// Réécriture de l'interface ONNX (onnx_io_type.h) : entrées et sorties retypées, Cast ajoutés
// autour d'un graphe intact, poids compris ; le modèle réécrit calcule comme l'original par
// l'interpréteur ; rien n'est réécrit quand l'interface est déjà dans le type.
#include "gpu/ir/onnx_io_type.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/ir/onnx_model.h"
#include "gpu/ir/protobuf.h"
#include "kernels/ref/half.h"
#include "test_util.h"

using ajp::RefusalCode;
using namespace ajp::gpu::ir;
using test::ModelBuilder;

namespace {

using Names = std::vector<std::string>;

// Le type cible d'un nœud Cast (attribut to).
std::int64_t cast_to(const onnx::Node& node) {
  REQUIRE(node.op_type == "Cast");
  REQUIRE(node.attributes.size() == 1);
  CHECK(node.attributes[0].name == "to");
  CHECK(node.attributes[0].type == onnx::AttributeType::Int);
  return node.attributes[0].i;
}

// Les noms des value_info (champ 13 de GraphProto), que le décodeur n'expose pas.
Names value_info_names(std::span<const std::byte> model) {
  Names names;
  proto::Reader reader(model);
  for (auto field = reader.next(); field && *field; field = reader.next()) {
    if ((*field)->number != 7) continue;
    proto::Reader graph((*field)->bytes);
    for (auto g = graph.next(); g && *g; g = graph.next()) {
      if ((*g)->number != 13) continue;
      proto::Reader info((*g)->bytes);
      for (auto i = info.next(); i && *i; i = info.next()) {
        if ((*i)->number == 1) names.push_back(*proto::as_string(**i));
      }
    }
  }
  return names;
}

}  // namespace

TEST_CASE("onnx : interface FLOAT16 autour d'un graphe FLOAT, poids et formes intacts", "[gpu_ir][onnx]") {
  ModelBuilder m;
  m.input("x", {1, 2, 2, 2})
      .weight("w", {1}, {2.0F})
      .node("Mul", {"x", "w"}, {"t"})
      .node("Relu", {"t"}, {"y"})
      .output("y");
  const std::vector<std::byte> original = m.bytes();
  auto rewritten = onnx::with_io_type(original, onnx::kFloat16);
  REQUIRE(rewritten);
  REQUIRE_FALSE(rewritten->empty());

  auto model = onnx::read_onnx(*rewritten);
  REQUIRE(model);
  const onnx::Graph& g = model->graph;
  REQUIRE(g.inputs.size() == 1);
  CHECK(g.inputs[0].name == "x");
  CHECK(g.inputs[0].elem_type == onnx::kFloat16);
  CHECK(g.inputs[0].dims == std::vector<std::int64_t>{1, 2, 2, 2});
  REQUIRE(g.outputs.size() == 1);
  CHECK(g.outputs[0].name == "y");
  CHECK(g.outputs[0].elem_type == onnx::kFloat16);
  REQUIRE(g.initializers.size() == 1);
  CHECK(g.initializers[0].data_type == onnx::kFloat);
  CHECK(g.initializers[0].floats == std::vector<float>{2.0F});
  REQUIRE(g.nodes.size() == 4);
  CHECK(g.nodes[0].inputs == Names{"x"});
  CHECK(g.nodes[0].outputs == Names{"x.f32"});
  CHECK(cast_to(g.nodes[0]) == onnx::kFloat);
  CHECK(g.nodes[1].op_type == "Mul");
  CHECK(g.nodes[1].inputs == Names{"x.f32", "w"});
  CHECK(g.nodes[2].op_type == "Relu");
  CHECK(g.nodes[2].outputs == Names{"y.f32"});
  CHECK(g.nodes[3].inputs == Names{"y.f32"});
  CHECK(g.nodes[3].outputs == Names{"y"});
  CHECK(cast_to(g.nodes[3]) == onnx::kFloat16);
  CHECK(model->opsets.size() == 1);

  // Même calcul que l'original, à l'arrondi F16 près de l'entrée et de la sortie.
  auto before = load_onnx(original);
  auto after = load_onnx(*rewritten);
  REQUIRE(before);
  REQUIRE(after);
  REQUIRE(infer_shapes(*before, {}));
  REQUIRE(infer_shapes(*after, {}));
  std::vector<float> values = test::sequence(8, -1.0F, 1.0F);
  for (float& v : values) v = ajp::kernels::ref::half_to_float(ajp::kernels::ref::float_to_half(v));
  const Value expected = test::execute1(*before, {{"x", test::floats({1, 2, 2, 2}, values)}});
  const Value got = test::execute1(*after, {{"x", test::floats({1, 2, 2, 2}, values, DataType::F16)}});
  CHECK(got.type == DataType::F16);
  CHECK(test::max_abs_diff(got.f, expected.f) <= 2e-3F);
}

TEST_CASE("onnx : rien à réécrire quand l'interface est déjà dans le type", "[gpu_ir][onnx]") {
  ModelBuilder m;
  m.input("x", {2}).node("Relu", {"x"}, {"y"}).output("y");
  auto same = onnx::with_io_type(m.bytes(), onnx::kFloat);
  REQUIRE(same);
  CHECK(same->empty());
  ModelBuilder half;
  half.input("x", {2}, onnx::kFloat16).node("Relu", {"x"}, {"y"}).output("y", onnx::kFloat16);
  auto same_half = onnx::with_io_type(half.bytes(), onnx::kFloat16);
  REQUIRE(same_half);
  CHECK(same_half->empty());
}

TEST_CASE("onnx : seules les entrées et sorties d'un autre type passent par un Cast", "[gpu_ir][onnx]") {
  ModelBuilder m;
  m.input("x", {2}, onnx::kFloat16)
      .node("Cast", {"x"}, {"t"}, {test::attr_int("to", onnx::kFloat)})
      .node("Relu", {"t"}, {"y"})
      .output("y");
  auto rewritten = onnx::with_io_type(m.bytes(), onnx::kFloat16);
  REQUIRE(rewritten);
  auto model = onnx::read_onnx(*rewritten);
  REQUIRE(model);
  const onnx::Graph& g = model->graph;
  CHECK(g.inputs[0].elem_type == onnx::kFloat16);
  CHECK(g.outputs[0].elem_type == onnx::kFloat16);
  REQUIRE(g.nodes.size() == 3);
  CHECK(g.nodes[0].inputs == Names{"x"});  // l'entrée garde son nom : déjà en FLOAT16
  CHECK(g.nodes[1].outputs == Names{"y.f32"});
  CHECK(g.nodes[2].op_type == "Cast");
  CHECK(g.nodes[2].outputs == Names{"y"});
}

TEST_CASE("onnx : une sortie relue par un nœud et sa value_info suivent le nom interne", "[gpu_ir][onnx]") {
  ModelBuilder m;
  m.input("x", {2}).node("Relu", {"x"}, {"y"}).node("Neg", {"y"}, {"z"}).output("y").output("z");
  m.graph().message(13, test::value_info("y", onnx::kFloat, {2}));
  m.graph().message(13, test::value_info("x", onnx::kFloat, {2}));
  auto rewritten = onnx::with_io_type(m.bytes(), onnx::kFloat16);
  REQUIRE(rewritten);
  auto model = onnx::read_onnx(*rewritten);
  REQUIRE(model);
  const onnx::Graph& g = model->graph;
  REQUIRE(g.nodes.size() == 5);
  CHECK(g.nodes[1].op_type == "Relu");
  CHECK(g.nodes[1].outputs == Names{"y.f32"});
  CHECK(g.nodes[2].op_type == "Neg");
  CHECK(g.nodes[2].inputs == Names{"y.f32"});
  CHECK(g.nodes[2].outputs == Names{"z.f32"});
  CHECK(g.nodes[3].inputs == Names{"y.f32"});
  CHECK(g.nodes[3].outputs == Names{"y"});
  CHECK(g.nodes[4].outputs == Names{"z"});
  CHECK(value_info_names(*rewritten) == Names{"y.f32", "x.f32"});
}

TEST_CASE("onnx : un nom interne déjà pris en reçoit un autre", "[gpu_ir][onnx]") {
  ModelBuilder m;
  m.input("x", {2}).node("Relu", {"x"}, {"x.f32"}).node("Neg", {"x.f32"}, {"y"}).output("y");
  auto rewritten = onnx::with_io_type(m.bytes(), onnx::kFloat16);
  REQUIRE(rewritten);
  auto model = onnx::read_onnx(*rewritten);
  REQUIRE(model);
  CHECK(model->graph.nodes[0].outputs == Names{"x.f322"});
  CHECK(model->graph.nodes[1].inputs == Names{"x.f322"});
  CHECK(model->graph.nodes[1].outputs == Names{"x.f32"});
}

TEST_CASE("onnx : interface non réécrite autour d'un sous-graphe ou d'un passage direct", "[gpu_ir][onnx]") {
  ModelBuilder loop;
  loop.input("x", {2})
      .node("If", {"x"}, {"y"},
            {test::ProtoWriter{}.bytes(1, "then_branch").message(6, test::ProtoWriter{}).varint(20, 5)})
      .output("y");
  auto refused = onnx::with_io_type(loop.bytes(), onnx::kFloat16);
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::FeatureUnavailable);

  ModelBuilder through;
  through.input("x", {2}).node("Relu", {"x"}, {"y"}).output("x").output("y");
  auto same_name = onnx::with_io_type(through.bytes(), onnx::kFloat16);
  REQUIRE_FALSE(same_name);
  CHECK(same_name.error().code == RefusalCode::FeatureUnavailable);

  auto bad_type = onnx::with_io_type(through.bytes(), onnx::kInt64);
  REQUIRE_FALSE(bad_type);
  CHECK(bad_type.error().code == RefusalCode::FeatureUnavailable);
}

TEST_CASE("onnx : dimensions symboliques et anonymes fixées sur les entrées", "[gpu_ir][onnx]") {
  // Une entrée à dimensions nommées (dim_param) et une autre à -1 sans nom, comme les RIFE de
  // vs-mlrt : les deux déclarent ensuite la forme demandée, le graphe et les poids restent.
  ModelBuilder m;
  m.input("x", {1, 3, -1, -1});
  test::ProtoWriter shape;
  for (const std::int64_t d : {1, 3, -1, -1}) shape.message(1, test::ProtoWriter{}.varint(1, d));
  m.graph().message(11, test::ProtoWriter{}.bytes(1, "z").message(
                            2, test::ProtoWriter{}.message(
                                   1, test::ProtoWriter{}.varint(1, onnx::kFloat16).message(2, shape))));
  m.weight("w", {1}, {2.0F}).node("Mul", {"x", "w"}, {"t"}).node("Add", {"t", "z"}, {"y"}).output("y");
  const std::vector<std::byte> original = m.bytes();
  auto before = onnx::read_onnx(original);
  REQUIRE(before);
  REQUIRE(before->graph.inputs.size() == 2);
  CHECK(before->graph.inputs[1].symbols == Names{"", "", "?", "?"});

  const std::vector<onnx::InputDims> fixed{{.name = "x", .dims = {1, 3, 540, 960}},
                                           {.name = "z", .dims = {1, 3, 540, 960}}};
  auto rewritten = onnx::with_input_dims(original, fixed);
  REQUIRE(rewritten);
  REQUIRE_FALSE(rewritten->empty());
  auto model = onnx::read_onnx(*rewritten);
  REQUIRE(model);
  const onnx::Graph& g = model->graph;
  REQUIRE(g.inputs.size() == 2);
  for (const onnx::ValueInfo& v : g.inputs) {
    CHECK(v.has_shape);
    CHECK(v.dims == std::vector<std::int64_t>{1, 3, 540, 960});
    CHECK(v.symbols == Names{"", "", "", ""});
  }
  CHECK(g.inputs[0].elem_type == onnx::kFloat);
  CHECK(g.inputs[1].elem_type == onnx::kFloat16);
  REQUIRE(g.nodes.size() == 2);
  CHECK(g.nodes[1].inputs == Names{"t", "z"});
  REQUIRE(g.initializers.size() == 1);
  CHECK(g.initializers[0].floats == std::vector<float>{2.0F});

  // Déjà fixées : rien à réécrire.
  auto again = onnx::with_input_dims(*rewritten, fixed);
  REQUIRE(again);
  CHECK(again->empty());
}

TEST_CASE("onnx : dimensions refusées quand elles contredisent le fichier", "[gpu_ir][onnx]") {
  ModelBuilder m;
  m.input("x", {1, 3, -1, -1}).node("Relu", {"x"}, {"y"}).output("y");
  const std::vector<std::byte> bytes = m.bytes();
  const auto refused = [&](onnx::InputDims dims) {
    auto out = onnx::with_input_dims(bytes, std::span(&dims, 1));
    REQUIRE_FALSE(out);
    CHECK(out.error().code == RefusalCode::InputMismatch);
  };
  refused({.name = "absente", .dims = {1, 3, 8, 8}});
  refused({.name = "x", .dims = {1, 3, 8}});     // rang
  refused({.name = "x", .dims = {1, 4, 8, 8}});  // dimension fixe du fichier
  refused({.name = "x", .dims = {1, 3, 0, 8}});  // non positive
  auto none = onnx::with_input_dims(bytes, {});
  REQUIRE(none);
  CHECK(none->empty());
}
