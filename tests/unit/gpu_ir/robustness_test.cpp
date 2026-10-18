// Entrées hostiles qui franchissent le décodeur : types croisés, constructions
// valides en ONNX mais au sens incertain, graphes construits à la main. Chacune
// doit être refusée avec son code, jamais lue hors des tampons ni corrigée en silence.
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <string_view>
#include <system_error>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/ir/memory_plan.h"
#include "test_util.h"

using ajp::RefusalCode;
using namespace ajp::gpu::ir;
using namespace ajp::gpu::ir::test;

namespace {

ModelBuilder conv_with_bias(std::string_view op, const ProtoWriter& bias) {
  ModelBuilder m;
  m.input("x", {1, 2, 3, 3}).weight("w", {2, 2, 1, 1}, {1, 0, 0, 1}).initializer(bias);
  m.node(op, {"x", "w", "b"}, {"y"}).output("y");
  return m;
}

// x [1, 1, 2, 2] -> Relu -> y : le plus petit graphe valide, à abîmer ensuite.
Graph relu_graph() {
  Graph g;
  g.opset = 17;
  g.tensors.push_back(
      {.name = "x", .kind = TensorKind::Input, .type = DataType::F32, .dims = {1, 1, 2, 2}, .known = true});
  g.tensors.push_back({.name = "y", .type = DataType::F32, .dims = {1, 1, 2, 2}, .known = true});
  g.nodes.push_back({.name = "relu", .op = OpKind::Relu, .inputs = {0}, .outputs = {1}});
  g.inputs = {0};
  g.outputs = {1};
  return g;
}

// Tous les points d'entrée publics refusent le graphe avec le même code.
void check_rejected_everywhere(Graph graph, RefusalCode code) {
  auto valid = validate(graph);
  REQUIRE_FALSE(valid);
  CHECK(valid.error().code == code);
  auto out = run(graph, std::vector<NamedValue>{{.name = "x", .value = random_floats({1, 1, 2, 2})}});
  REQUIRE_FALSE(out);
  CHECK(out.error().code == code);
  auto plan = plan_memory(graph);
  REQUIRE_FALSE(plan);
  CHECK(plan.error().code == code);
  auto shapes = infer_shapes(graph);
  REQUIRE_FALSE(shapes);
  CHECK(shapes.error().code == code);
}

}  // namespace

TEST_CASE("robustesse : biais d'un autre type que l'entrée refusé", "[gpu_ir][robust]") {
  // Un biais I64 était lu comme des float absents : lecture hors du tampon.
  for (const std::string_view op : {"Conv", "ConvTranspose"}) {
    INFO(op);
    const std::vector<std::int64_t> ints{1, 2};
    CHECK(refusal(conv_with_bias(op, tensor_i64("b", {2}, ints))).code == RefusalCode::GraphInvalid);
    const std::vector<float> halves{0.5F, -0.5F};
    CHECK(refusal(conv_with_bias(op, tensor_f16("b", {2}, halves))).code == RefusalCode::GraphInvalid);
    const std::vector<float> same{0.5F, -0.5F};
    CHECK(load_onnx(conv_with_bias(op, tensor_f32("b", {2}, same)).bytes()).has_value());
  }
}

TEST_CASE("robustesse : AveragePool SAME au pas plus grand que le noyau refusé", "[gpu_ir][robust]") {
  // Marge SAME négative : ONNX Runtime décale la fenêtre, la référence non. Conv la ramène à 0.
  auto model = [](std::string_view op) {
    ModelBuilder m;
    m.input("x", {1, 1, 5, 5}).weight("w", {1, 1, 1, 1}, {1});
    const std::initializer_list<ProtoWriter> attrs{attr_string("auto_pad", "SAME_UPPER"),
                                                   attr_ints("strides", {3, 3}),
                                                   attr_ints("kernel_shape", {1, 1})};
    if (op == "Conv") {
      m.node(op, {"x", "w"}, {"y"}, attrs);
    } else {
      m.node(op, {"x"}, {"y"}, attrs);
    }
    return m.output("y");
  };
  CHECK(refusal(model("AveragePool")).code == RefusalCode::OnnxUnsupported);
  const Value x = floats({1, 1, 5, 5}, sequence(25, 0, 25));
  const Value y = execute1(build(model("Conv")), {{.name = "x", .value = x}});
  REQUIRE(y.dims == Dims{1, 1, 2, 2});
  CHECK(y.f == std::vector<float>{x.f[0], x.f[3], x.f[15], x.f[18]});
}

TEST_CASE("robustesse : Slice à rebours jusqu'à INT_MAX refusé, INT_MIN accepté", "[gpu_ir][robust]") {
  auto model = [](std::int64_t end) {
    ModelBuilder m;
    m.input("x", {1, 4}).ints("starts", {-1}).ints("ends", {end}).ints("axes", {1}).ints("steps", {-1});
    return m.node("Slice", {"x", "starts", "ends", "axes", "steps"}, {"y"}).output("y");
  };
  CHECK(refusal(model(std::numeric_limits<std::int64_t>::max())).code == RefusalCode::OnnxUnsupported);
  CHECK(refusal(model(std::numeric_limits<std::int32_t>::max())).code == RefusalCode::OnnxUnsupported);
  const Value x = floats({1, 4}, {1, 2, 3, 4});
  const Value y =
      execute1(build(model(std::numeric_limits<std::int64_t>::min())), {{.name = "x", .value = x}});
  CHECK(y.f == std::vector<float>{4, 3, 2, 1});
}

TEST_CASE("robustesse : Pad à l'axe répété refusé", "[gpu_ir][robust]") {
  ModelBuilder m(18);
  m.input("x", {1, 1, 2, 2}).ints("pads", {1, 1, 1, 1}).ints("axes", {3, -1});
  m.node("Pad", {"x", "pads", "", "axes"}, {"y"}).output("y");
  CHECK(refusal(m).code == RefusalCode::GraphInvalid);
}

TEST_CASE("robustesse : règles de Split selon l'opset", "[gpu_ir][robust]") {
  auto model = [](std::int64_t opset, std::int64_t extent, std::initializer_list<ProtoWriter> attrs) {
    ModelBuilder m(opset);
    m.input("x", {1, extent}).node("Split", {"x"}, {"a", "b"}, attrs).output("a").output("b");
    return m;
  };
  const ProtoWriter axis = attr_int("axis", 1);
  // Avant l'opset 18, sans tailles, les parts sont strictement égales.
  CHECK(refusal(model(13, 5, {axis})).code == RefusalCode::ShapeInvalid);
  Graph even = build(model(13, 6, {axis}));
  CHECK(even.tensors[even.outputs[1]].dims == Dims{1, 3});
  // Opset 18 : num_outputs arrondit au-dessus, la dernière part est plus petite.
  Graph ceil = build(model(18, 5, {axis, attr_int("num_outputs", 2)}));
  CHECK(ceil.tensors[ceil.outputs[0]].dims == Dims{1, 3});
  CHECK(ceil.tensors[ceil.outputs[1]].dims == Dims{1, 2});
  CHECK(refusal(model(18, 5, {axis, attr_int("num_outputs", 3)})).code == RefusalCode::GraphInvalid);

  ModelBuilder both(18);
  both.input("x", {1, 5}).ints("split", {3, 2});
  both.node("Split", {"x", "split"}, {"a", "b"}, {axis, attr_int("num_outputs", 2)}).output("a").output("b");
  CHECK(refusal(both).code == RefusalCode::GraphInvalid);
}

TEST_CASE("robustesse : GridSample absent avant l'opset 16", "[gpu_ir][robust]") {
  auto model = [](std::int64_t opset) {
    ModelBuilder m(opset);
    m.input("x", {1, 1, 2, 2}).input("grid", {1, 1, 1, 2});
    return m.node("GridSample", {"x", "grid"}, {"y"}).output("y");
  };
  CHECK(refusal(model(15)).code == RefusalCode::OnnxUnsupported);
  CHECK(load_onnx(model(16).bytes()).has_value());
}

TEST_CASE("robustesse : attribut de Constant au type contraire à son nom", "[gpu_ir][robust]") {
  // value_float déclaré Ints : le float lu serait 0 en silence.
  auto model = [](const ProtoWriter& attr) {
    ModelBuilder m;
    m.input("x", {1}).node("Constant", {}, {"k"}, {attr}).node("Mul", {"x", "k"}, {"y"}).output("y");
    return m;
  };
  CHECK(refusal(model(attr_ints("value_float", {3}))).code == RefusalCode::GraphInvalid);
  CHECK(refusal(model(attr_float("value_ints", 3))).code == RefusalCode::GraphInvalid);
  CHECK(refusal(model(attr_int("value", 3))).code == RefusalCode::GraphInvalid);
  const Value y =
      execute1(build(model(attr_float("value_float", 3))), {{.name = "x", .value = floats({1}, {2})}});
  CHECK(y.f == std::vector<float>{6});
}

TEST_CASE("robustesse : graphe construit à la main et incohérent", "[gpu_ir][robust]") {
  REQUIRE(validate(relu_graph()));
  REQUIRE(plan_memory(relu_graph()));

  Graph outside = relu_graph();
  outside.inputs = {7};
  check_rejected_everywhere(outside, RefusalCode::GraphInvalid);

  Graph read_outside = relu_graph();
  read_outside.nodes[0].inputs = {9};
  check_rejected_everywhere(read_outside, RefusalCode::GraphInvalid);

  // Constante déclarée [1, 1, 2, 2] mais porteuse d'un seul élément.
  Graph short_constant = relu_graph();
  short_constant.tensors.push_back({.name = "k",
                                    .kind = TensorKind::Constant,
                                    .dims = {1, 1, 2, 2},
                                    .known = true,
                                    .value = floats({1, 1, 2, 2}, {1})});
  short_constant.nodes[0].inputs = {2};
  check_rejected_everywhere(short_constant, RefusalCode::GraphInvalid);

  Graph wrong_type = relu_graph();
  wrong_type.tensors.push_back({.name = "k",
                                .kind = TensorKind::Constant,
                                .dims = {2},
                                .known = true,
                                .value = Value{.type = DataType::I64, .dims = {2}, .i = {1, 2}}});
  check_rejected_everywhere(wrong_type, RefusalCode::GraphInvalid);

  // Entrée hors de l'interface : jamais fournie, jamais formée.
  Graph hidden_input = relu_graph();
  hidden_input.tensors.push_back(
      {.name = "z", .kind = TensorKind::Input, .dims = {1, 1, 2, 2}, .known = true});
  check_rejected_everywhere(hidden_input, RefusalCode::GraphInvalid);

  Graph written_twice = relu_graph();
  written_twice.nodes.push_back(written_twice.nodes[0]);
  check_rejected_everywhere(written_twice, RefusalCode::GraphInvalid);
}

TEST_CASE("robustesse : plan mémoire sans formes connues refusé", "[gpu_ir][robust]") {
  ModelBuilder m;
  m.input("x", {1, 3, -1, -1}).node("Relu", {"x"}, {"y"}).output("y");
  auto graph = load_onnx(m.bytes());
  REQUIRE(graph);
  auto plan = plan_memory(*graph);
  REQUIRE_FALSE(plan);
  CHECK(plan.error().code == RefusalCode::ShapeInvalid);
}

TEST_CASE("robustesse : entrée donnée hors de la forme déclarée", "[gpu_ir][robust]") {
  ModelBuilder m;
  m.input("x", {1, 3, -1, -1}).node("Relu", {"x"}, {"y"}).output("y");
  auto graph = load_onnx(m.bytes());
  REQUIRE(graph);
  auto ok = run(*graph, std::vector<NamedValue>{{.name = "x", .value = random_floats({1, 3, 2, 5})}});
  CHECK(ok.has_value());
  // Quatre canaux pour trois déclarés : la dimension connue doit être respectée.
  auto channels = run(*graph, std::vector<NamedValue>{{.name = "x", .value = random_floats({1, 4, 2, 2})}});
  REQUIRE_FALSE(channels);
  CHECK(channels.error().code == RefusalCode::ShapeInvalid);

  // Rang inconnu : (2^24)^4 éléments annoncés sans données. Le produit déborde à 0
  // et égalait la taille des données vides ; la borne le refuse avant tout calcul.
  ModelBuilder free_rank;
  free_rank.graph().message(11, value_info("x", onnx::kFloat, {}, false));
  free_rank.node("Relu", {"x"}, {"y"}).output("y");
  auto unranked = load_onnx(free_rank.bytes());
  REQUIRE(unranked);
  constexpr std::int64_t kHuge = std::int64_t{1} << 24;
  auto huge = run(*unranked,
                  std::vector<NamedValue>{{.name = "x", .value = floats({kHuge, kHuge, kHuge, kHuge}, {})}});
  REQUIRE_FALSE(huge);
  CHECK(huge.error().code == RefusalCode::GraphInvalid);
}

TEST_CASE("robustesse : lecture d'un modèle sur disque", "[gpu_ir][robust]") {
  const std::filesystem::path dir = std::filesystem::temp_directory_path();
  const std::filesystem::path path = dir / "ajp_gpu_ir_robustness_model.onnx";
  ModelBuilder m;
  m.input("x", {1, 1, 2, 2}).node("Relu", {"x"}, {"y"}).output("y");
  const std::vector<std::byte> bytes = m.bytes();
  {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    REQUIRE(file.good());
  }
  auto graph = load_onnx_file(path);
  std::error_code ignored;
  std::filesystem::remove(path, ignored);
  REQUIRE(graph);
  CHECK(graph->nodes.size() == 1);

  auto missing = load_onnx_file(dir / "ajp_gpu_ir_modele_absent.onnx");
  REQUIRE_FALSE(missing);
  CHECK(missing.error().code == RefusalCode::OnnxMalformed);
  CHECK_FALSE(load_onnx_file(dir).has_value());  // un dossier n'est pas un modèle
}
