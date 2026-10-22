// Réécriture de l'interface ONNX (onnx_io_type.h) : entrées et sorties retypées, Cast ajoutés
// autour d'un graphe intact, poids compris ; le modèle réécrit calcule comme l'original par
// l'interpréteur ; rien n'est réécrit quand l'interface est déjà dans le type. Les métadonnées
// du modèle (with_metadata).
#include "gpu/ir/onnx_io_type.h"

#include <cstdint>
#include <span>
#include <string>
#include <utility>
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

namespace {

// Une entrée dont les dimensions négatives sont écrites -1, sans nom (exports RIFE de vs-mlrt).
test::ProtoWriter unnamed_input(std::string_view name, const std::vector<std::int64_t>& dims) {
  test::ProtoWriter shape;
  for (const std::int64_t d : dims) shape.message(1, test::ProtoWriter{}.varint(1, d < 0 ? -1 : d));
  test::ProtoWriter tensor_type;
  tensor_type.varint(1, onnx::kFloat16).message(2, shape);
  return test::ProtoWriter{}.bytes(1, name).message(2, test::ProtoWriter{}.message(1, tensor_type));
}

}  // namespace

TEST_CASE("onnx : entrées fixées à la forme voulue, dimensions libres nommées ou non", "[gpu_ir][onnx]") {
  ModelBuilder m;
  m.input("x", {1, 2, -1, -1}).weight("w", {1}, {2.0F}).node("Mul", {"x", "w"}, {"t"});
  m.graph().message(11, unnamed_input("y", {1, 2, -1, -1}));
  m.node("Add", {"t", "y"}, {"z"}).output("z");
  const std::vector<std::byte> original = m.bytes();
  const std::vector<std::int64_t> shape{1, 2, 4, 6};
  const std::vector<onnx::InputDims> fixed{{.name = "x", .dims = shape}, {.name = "y", .dims = shape}};
  auto rewritten = onnx::with_input_dims(original, fixed);
  REQUIRE(rewritten);
  REQUIRE_FALSE(rewritten->empty());

  auto before = onnx::read_onnx(original);
  auto model = onnx::read_onnx(*rewritten);
  REQUIRE(before);
  REQUIRE(model);
  CHECK(before->graph.inputs[0].symbols == Names{"", "", "d2", "d3"});
  CHECK(before->graph.inputs[1].symbols == Names{"", "", "?", "?"});
  const onnx::Graph& g = model->graph;
  REQUIRE(g.inputs.size() == 2);
  for (const onnx::ValueInfo& v : g.inputs) {
    CHECK(v.dims == shape);
    CHECK(v.symbols == Names{"", "", "", ""});
  }
  CHECK(g.inputs[0].elem_type == onnx::kFloat);
  CHECK(g.inputs[1].elem_type == onnx::kFloat16);
  REQUIRE(g.nodes.size() == before->graph.nodes.size());
  for (std::size_t k = 0; k < g.nodes.size(); ++k) {
    CHECK(g.nodes[k].op_type == before->graph.nodes[k].op_type);
    CHECK(g.nodes[k].inputs == before->graph.nodes[k].inputs);
  }
  REQUIRE(g.initializers.size() == 1);
  CHECK(g.initializers[0].floats == std::vector<float>{2.0F});

  // Déjà fixées : rien à réécrire.
  auto again = onnx::with_input_dims(*rewritten, fixed);
  REQUIRE(again);
  CHECK(again->empty());
}

TEST_CASE("onnx : forme d'entrée refusée quand elle contredit le fichier", "[gpu_ir][onnx]") {
  ModelBuilder m;
  m.input("x", {1, 3, -1, -1}).node("Relu", {"x"}, {"y"}).output("y");
  const std::vector<std::byte> bytes = m.bytes();
  const auto refused = [&](std::string_view name, std::vector<std::int64_t> dims) {
    const std::vector<onnx::InputDims> fixed{{.name = name, .dims = dims}};
    auto out = onnx::with_input_dims(bytes, fixed);
    REQUIRE_FALSE(out);
    return out.error().code;
  };
  CHECK(refused("x", {1, 4, 8, 8}) == RefusalCode::InputMismatch);  // 3 canaux dans le fichier
  CHECK(refused("x", {1, 3, 8}) == RefusalCode::InputMismatch);     // autre rang
  CHECK(refused("x", {1, 3, 8, 0}) == RefusalCode::InputMismatch);  // forme voulue encore libre
  CHECK(refused("z", {1, 3, 8, 8}) == RefusalCode::InputMismatch);  // pas une entrée
}

namespace {

// Les métadonnées du ModelProto (champ 14 : clé 1, valeur 2), que le décodeur n'expose pas.
std::vector<std::pair<std::string, std::string>> metadata_of(std::span<const std::byte> model) {
  std::vector<std::pair<std::string, std::string>> out;
  proto::Reader reader(model);
  for (auto field = reader.next(); field && *field; field = reader.next()) {
    if ((*field)->number != 14) continue;
    std::pair<std::string, std::string> entry;
    proto::Reader inner((*field)->bytes);
    for (auto f = inner.next(); f && *f; f = inner.next()) {
      if ((*f)->number == 1) entry.first = *proto::as_string(**f);
      if ((*f)->number == 2) entry.second = *proto::as_string(**f);
    }
    out.push_back(std::move(entry));
  }
  return out;
}

}  // namespace

TEST_CASE("onnx : une entrée de métadonnées posée, remplacée, ou déjà là", "[gpu_ir][onnx]") {
  ModelBuilder m;
  m.input("x", {1, 2, 2, 2}).weight("w", {1}, {2.0F}).node("Mul", {"x", "w"}, {"y"}).output("y");
  const std::vector<std::byte> original = m.bytes();
  auto keyed = onnx::with_metadata(original, "COREML_CACHE_KEY", "ajp");
  REQUIRE(keyed);
  REQUIRE_FALSE(keyed->empty());
  using Entries = std::vector<std::pair<std::string, std::string>>;
  CHECK(metadata_of(*keyed) == Entries{{"COREML_CACHE_KEY", "ajp"}});
  // Le graphe et ses poids intacts : le modèle calcule comme avant.
  auto before = onnx::read_onnx(original);
  auto after = onnx::read_onnx(*keyed);
  REQUIRE(before);
  REQUIRE(after);
  CHECK(after->graph.nodes.size() == before->graph.nodes.size());
  REQUIRE(after->graph.initializers.size() == 1);
  CHECK(after->graph.initializers[0].floats == std::vector<float>{2.0F});
  // Déjà là : rien à réécrire.
  auto again = onnx::with_metadata(*keyed, "COREML_CACHE_KEY", "ajp");
  REQUIRE(again);
  CHECK(again->empty());
  // Une autre valeur remplace la première ; une autre clé s'ajoute à côté.
  auto replaced = onnx::with_metadata(*keyed, "COREML_CACHE_KEY", "autre");
  REQUIRE(replaced);
  CHECK(metadata_of(*replaced) == Entries{{"COREML_CACHE_KEY", "autre"}});
  auto both = onnx::with_metadata(*replaced, "producteur", "ajp");
  REQUIRE(both);
  CHECK(metadata_of(*both) == Entries{{"COREML_CACHE_KEY", "autre"}, {"producteur", "ajp"}});
}

namespace {

// Les types des value_info (champ 13 de GraphProto), que le décodeur n'expose pas.
std::vector<std::int64_t> value_info_types(std::span<const std::byte> model) {
  std::vector<std::int64_t> types;
  const auto inside = [](std::span<const std::byte> bytes, std::uint32_t number) {
    proto::Reader reader(bytes);
    for (auto f = reader.next(); f && *f; f = reader.next()) {
      if ((*f)->number == number) return (*f)->bytes;
    }
    return std::span<const std::byte>{};
  };
  proto::Reader reader(model);
  for (auto field = reader.next(); field && *field; field = reader.next()) {
    if ((*field)->number != 7) continue;
    proto::Reader graph((*field)->bytes);
    for (auto g = graph.next(); g && *g; g = graph.next()) {
      if ((*g)->number != 13) continue;
      const std::span<const std::byte> tensor = inside(inside((*g)->bytes, 2), 1);
      proto::Reader fields(tensor);
      for (auto e = fields.next(); e && *e; e = fields.next()) {
        if ((*e)->number == 1) types.push_back(static_cast<std::int64_t>((*e)->scalar));
      }
    }
  }
  return types;
}

}  // namespace

TEST_CASE("onnx : un modèle FLOAT16 entier en FLOAT calcule comme l'original", "[gpu_ir][onnx]") {
  using ajp::kernels::ref::float_to_half;
  using ajp::kernels::ref::half_to_float;
  // Poids en raw_data et en int32_data, une constante, une value_info et un Cast vers FLOAT16.
  ModelBuilder m;
  m.input("x", {1, 2, 2, 2}, onnx::kFloat16)
      .initializer(test::tensor_f16("w", {1}, std::vector<float>{2.5F}))
      .initializer(test::tensor_header("b", onnx::kFloat16, std::vector<std::int64_t>{1})
                       .packed_varints(5, std::vector<std::int64_t>{float_to_half(0.5F)}))
      .node("Mul", {"x", "w"}, {"t"})
      .node("Add", {"t", "b"}, {"u"})
      .node("Constant", {}, {"c"},
            {test::attr_tensor("value", test::tensor_f16("c", {1}, std::vector<float>{1.0F}))})
      .node("Sub", {"u", "c"}, {"v"})
      .node("Cast", {"v"}, {"y"}, {test::attr_int("to", onnx::kFloat16)})
      .output("y", onnx::kFloat16);
  m.graph().message(13, test::value_info("t", onnx::kFloat16, {1, 2, 2, 2}));
  const std::vector<std::byte> original = m.bytes();
  auto converted = onnx::with_float32(original);
  REQUIRE(converted);
  REQUIRE_FALSE(converted->empty());

  auto model = onnx::read_onnx(*converted);
  REQUIRE(model);
  const onnx::Graph& g = model->graph;
  REQUIRE(g.inputs.size() == 1);
  CHECK(g.inputs[0].elem_type == onnx::kFloat);
  CHECK(g.inputs[0].dims == std::vector<std::int64_t>{1, 2, 2, 2});
  REQUIRE(g.outputs.size() == 1);
  CHECK(g.outputs[0].elem_type == onnx::kFloat);
  REQUIRE(g.initializers.size() == 2);
  CHECK(g.initializers[0].data_type == onnx::kFloat);
  CHECK(g.initializers[0].floats == std::vector<float>{2.5F});
  CHECK(g.initializers[1].data_type == onnx::kFloat);
  CHECK(g.initializers[1].floats == std::vector<float>{0.5F});
  REQUIRE(g.nodes.size() == 5);
  REQUIRE(g.nodes[2].attributes.size() == 1);
  REQUIRE(g.nodes[2].attributes[0].tensors.size() == 1);
  CHECK(g.nodes[2].attributes[0].tensors[0].data_type == onnx::kFloat);
  CHECK(g.nodes[2].attributes[0].tensors[0].floats == std::vector<float>{1.0F});
  CHECK(cast_to(g.nodes[4]) == onnx::kFloat);
  CHECK(value_info_types(*converted) == std::vector<std::int64_t>{onnx::kFloat});
  // Déjà en FLOAT : rien à réécrire.
  auto again = onnx::with_float32(*converted);
  REQUIRE(again);
  CHECK(again->empty());

  // Le même calcul, à l'arrondi F16 près des résultats intermédiaires de l'original.
  auto before = load_onnx(original);
  auto after = load_onnx(*converted);
  REQUIRE(before);
  REQUIRE(after);
  REQUIRE(infer_shapes(*before, {}));
  REQUIRE(infer_shapes(*after, {}));
  std::vector<float> values = test::sequence(8, -1.0F, 1.0F);
  for (float& v : values) v = half_to_float(float_to_half(v));
  const Value expected = test::execute1(*before, {{"x", test::floats({1, 2, 2, 2}, values, DataType::F16)}});
  const Value got = test::execute1(*after, {{"x", test::floats({1, 2, 2, 2}, values)}});
  CHECK(got.type == DataType::F32);
  CHECK(test::max_abs_diff(got.f, expected.f) <= 2e-3F);

  // Puis l'interface remise en FLOAT16 (la chaîne), le calcul restant en FLOAT.
  auto typed = onnx::with_io_type(*converted, onnx::kFloat16);
  REQUIRE(typed);
  REQUIRE_FALSE(typed->empty());
  auto interfaced = load_onnx(*typed);
  REQUIRE(interfaced);
  REQUIRE(infer_shapes(*interfaced, {}));
  const Value half = test::execute1(*interfaced, {{"x", test::floats({1, 2, 2, 2}, values, DataType::F16)}});
  CHECK(half.type == DataType::F16);
  CHECK(test::max_abs_diff(half.f, expected.f) <= 2e-3F);
}

TEST_CASE("onnx : rien à convertir en FLOAT, ou un sous-graphe refusé", "[gpu_ir][onnx]") {
  ModelBuilder plain;
  plain.input("x", {2}).weight("w", {1}, {2.0F}).node("Mul", {"x", "w"}, {"y"}).output("y");
  auto same = onnx::with_float32(plain.bytes());
  REQUIRE(same);
  CHECK(same->empty());

  ModelBuilder branch;
  branch.input("x", {2}, onnx::kFloat16)
      .node("If", {"x"}, {"y"},
            {test::ProtoWriter{}.bytes(1, "then_branch").message(6, test::ProtoWriter{}).varint(20, 5)})
      .output("y", onnx::kFloat16);
  auto refused = onnx::with_float32(branch.bytes());
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::FeatureUnavailable);
}
