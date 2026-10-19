#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/ir/onnx_model.h"
#include "onnx_builder.h"

using ajp::RefusalCode;
using namespace ajp::gpu::ir;
using test::ModelBuilder;
using test::ProtoWriter;

namespace {

// Un modèle valide dont le seul initialiseur est tensor.
std::vector<std::byte> model_with(const ProtoWriter& tensor) {
  ModelBuilder m;
  m.input("x", {1}).initializer(tensor).node("Add", {"x", "w"}, {"y"}).output("y");
  return m.bytes();
}

std::expected<onnx::Tensor, ajp::Refusal> read_initializer(const ProtoWriter& tensor) {
  auto model = onnx::read_onnx(model_with(tensor));
  if (!model) return std::unexpected(model.error());
  REQUIRE(model->graph.initializers.size() == 1);
  return model->graph.initializers[0];
}

RefusalCode refusal_of(const ProtoWriter& tensor) {
  auto read = read_initializer(tensor);
  REQUIRE_FALSE(read);
  return read.error().code;
}

}  // namespace

TEST_CASE("onnx : ModelProto complet relu champ par champ", "[gpu_ir][onnx]") {
  ModelBuilder m(13);
  m.input("img", {1, 3, -1, -1})
      .weight("w", {2, 3, 1, 1}, {1, 2, 3, 4, 5, 6})
      .node("Conv", {"img", "w", ""}, {"y"},
            {test::attr_ints("strides", {1, 1}), test::attr_int("group", 1), test::attr_float("alpha", 0.25F),
             test::attr_string("auto_pad", "NOTSET"), test::attr_floats("scales", {1.5F, 2.0F})})
      .output("y");
  m.graph().bytes(2, "réseau");
  auto model = onnx::read_onnx(m.bytes());
  REQUIRE(model);
  CHECK(model->ir_version == 8);
  CHECK(model->producer_name == "ajp-test");
  REQUIRE(model->opsets.size() == 1);
  CHECK(model->opsets[0].domain.empty());
  CHECK(model->opsets[0].version == 13);

  const onnx::Graph& g = model->graph;
  CHECK(g.name == "réseau");
  REQUIRE(g.inputs.size() == 1);
  CHECK(g.inputs[0].name == "img");
  CHECK(g.inputs[0].elem_type == onnx::kFloat);
  CHECK(g.inputs[0].has_shape);
  CHECK(g.inputs[0].dims == std::vector<std::int64_t>{1, 3, -1, -1});
  CHECK(g.inputs[0].symbols == std::vector<std::string>{"", "", "d2", "d3"});
  REQUIRE(g.outputs.size() == 1);
  CHECK_FALSE(g.outputs[0].has_shape);

  REQUIRE(g.nodes.size() == 1);
  const onnx::Node& conv = g.nodes[0];
  CHECK(conv.op_type == "Conv");
  CHECK(conv.inputs == std::vector<std::string>{"img", "w", ""});
  CHECK(conv.outputs == std::vector<std::string>{"y"});
  REQUIRE(conv.attributes.size() == 5);
  CHECK(conv.attributes[0].type == onnx::AttributeType::Ints);
  CHECK(conv.attributes[0].ints == std::vector<std::int64_t>{1, 1});
  CHECK(conv.attributes[1].i == 1);
  CHECK(conv.attributes[2].f == 0.25F);
  CHECK(conv.attributes[3].s == "NOTSET");
  CHECK(conv.attributes[4].floats == std::vector<float>{1.5F, 2.0F});

  REQUIRE(g.initializers.size() == 1);
  CHECK(g.initializers[0].name == "w");
  CHECK(g.initializers[0].dims == std::vector<std::int64_t>{2, 3, 1, 1});
  CHECK(g.initializers[0].floats == std::vector<float>{1, 2, 3, 4, 5, 6});
}

TEST_CASE("onnx : FLOAT en float_data, élément par élément ou empaqueté", "[gpu_ir][onnx]") {
  ProtoWriter t = test::tensor_header("w", onnx::kFloat, std::vector<std::int64_t>{3});
  t.f32(4, 1.0F).packed_floats(4, std::vector<float>{2.0F, -0.5F});
  auto read = read_initializer(t);
  REQUIRE(read);
  CHECK(read->floats == std::vector<float>{1.0F, 2.0F, -0.5F});
}

TEST_CASE("onnx : FLOAT16 brut et en int32_data, conversion exacte", "[gpu_ir][onnx]") {
  const std::vector<float> values{1.0F, -2.5F, 65504.0F, 6.103515625e-05F};
  auto raw = read_initializer(test::tensor_f16("w", {4}, values));
  REQUIRE(raw);
  CHECK(raw->data_type == onnx::kFloat16);
  CHECK(raw->floats == values);

  ProtoWriter words = test::tensor_header("w", onnx::kFloat16, std::vector<std::int64_t>{2});
  words.packed_varints(5, std::vector<std::int64_t>{0x3C00, 0xC100});  // 1.0 et -2.5
  auto typed = read_initializer(words);
  REQUIRE(typed);
  CHECK(typed->floats == std::vector<float>{1.0F, -2.5F});

  ProtoWriter wide = test::tensor_header("w", onnx::kFloat16, std::vector<std::int64_t>{1});
  wide.packed_varints(5, std::vector<std::int64_t>{0x10000});
  CHECK(refusal_of(wide) == RefusalCode::OnnxMalformed);
}

TEST_CASE("onnx : INT64 et INT32 bruts ou typés", "[gpu_ir][onnx]") {
  std::string raw64;
  ProtoWriter::append_le(raw64, static_cast<std::uint64_t>(-3), 8);
  ProtoWriter::append_le(raw64, 1ULL << 40U, 8);
  auto i64 =
      read_initializer(test::tensor_header("w", onnx::kInt64, std::vector<std::int64_t>{2}).bytes(9, raw64));
  REQUIRE(i64);
  CHECK(i64->ints == std::vector<std::int64_t>{-3, std::int64_t{1} << 40});

  auto typed = read_initializer(test::tensor_i64("w", {3}, std::vector<std::int64_t>{-1, 0, 7}));
  REQUIRE(typed);
  CHECK(typed->ints == std::vector<std::int64_t>{-1, 0, 7});

  std::string raw32;
  ProtoWriter::append_le(raw32, 0xFFFFFFFEU, 4);
  auto i32 =
      read_initializer(test::tensor_header("w", onnx::kInt32, std::vector<std::int64_t>{1}).bytes(9, raw32));
  REQUIRE(i32);
  CHECK(i32->ints == std::vector<std::int64_t>{-2});

  ProtoWriter words = test::tensor_header("w", onnx::kInt32, std::vector<std::int64_t>{2});
  words.varint(5, 4).varint(5, -9);
  auto i32_typed = read_initializer(words);
  REQUIRE(i32_typed);
  CHECK(i32_typed->ints == std::vector<std::int64_t>{4, -9});
}

TEST_CASE("onnx : poids externes refusés avec leur propre refus", "[gpu_ir][onnx]") {
  ProtoWriter external = test::tensor_header("w", onnx::kFloat, std::vector<std::int64_t>{1});
  external.message(13, ProtoWriter{}.bytes(1, "location").bytes(2, "poids.bin")).varint(14, 1);
  auto read = read_initializer(external);
  REQUIRE_FALSE(read);
  CHECK(read.error().code == RefusalCode::OnnxExternalData);
  CHECK(read.error().detail.find('w') != std::string::npos);

  ProtoWriter location_only = test::tensor_header("w", onnx::kFloat, std::vector<std::int64_t>{1});
  location_only.varint(14, 1);
  CHECK(refusal_of(location_only) == RefusalCode::OnnxExternalData);
}

TEST_CASE("onnx : données incohérentes avec la forme refusées", "[gpu_ir][onnx]") {
  // Trop peu d'octets bruts.
  CHECK(refusal_of(test::tensor_header("w", onnx::kFloat, std::vector<std::int64_t>{2}).bytes(9, "abcd")) ==
        RefusalCode::OnnxMalformed);
  // Octets bruts non multiples de la taille d'un élément.
  CHECK(refusal_of(test::tensor_header("w", onnx::kFloat16, std::vector<std::int64_t>{1}).bytes(9, "abc")) ==
        RefusalCode::OnnxMalformed);
  // Trop de valeurs typées.
  ProtoWriter many = test::tensor_header("w", onnx::kFloat, std::vector<std::int64_t>{1});
  many.packed_floats(4, std::vector<float>{1.0F, 2.0F});
  CHECK(refusal_of(many) == RefusalCode::OnnxMalformed);
  // Brutes et typées à la fois.
  ProtoWriter both = test::tensor_f32("w", {1}, std::vector<float>{1.0F});
  both.f32(4, 1.0F);
  CHECK(refusal_of(both) == RefusalCode::OnnxMalformed);
  // Valeurs dans le champ d'un autre type.
  ProtoWriter wrong = test::tensor_header("w", onnx::kInt64, std::vector<std::int64_t>{1});
  wrong.f32(4, 1.0F);
  CHECK(refusal_of(wrong) == RefusalCode::OnnxMalformed);
  // Dimension négative, et forme dont le produit déborde : refusées sans allocation.
  CHECK(refusal_of(test::tensor_header("w", onnx::kFloat, std::vector<std::int64_t>{-1})) ==
        RefusalCode::OnnxMalformed);
  CHECK(refusal_of(test::tensor_header("w", onnx::kFloat, std::vector<std::int64_t>{1 << 20, 1 << 20})) ==
        RefusalCode::OnnxMalformed);
}

TEST_CASE("onnx : types et constructions hors du sous-ensemble refusés", "[gpu_ir][onnx]") {
  constexpr std::int32_t kDouble = 11;
  CHECK(refusal_of(test::tensor_header("w", kDouble, std::vector<std::int64_t>{1}).bytes(9, "12345678")) ==
        RefusalCode::OnnxUnsupported);
  ProtoWriter segment = test::tensor_f32("w", {1}, std::vector<float>{1.0F});
  segment.message(3, ProtoWriter{}.varint(1, 0).varint(2, 1));
  CHECK(refusal_of(segment) == RefusalCode::OnnxUnsupported);

  ModelBuilder sparse;
  sparse.input("x", {1}).node("Relu", {"x"}, {"y"}).output("y");
  sparse.graph().message(15, ProtoWriter{});
  CHECK(onnx::read_onnx(sparse.bytes()).error().code == RefusalCode::OnnxUnsupported);

  // Entrée d'un type non tenseur (séquence : TypeProto champ 4).
  ModelBuilder sequence;
  sequence.node("Relu", {"x"}, {"y"}).output("y");
  sequence.graph().message(11,
                           ProtoWriter{}.bytes(1, "x").message(2, ProtoWriter{}.message(4, ProtoWriter{})));
  CHECK(onnx::read_onnx(sequence.bytes()).error().code == RefusalCode::OnnxUnsupported);

  ModelBuilder function_ref;
  function_ref.input("x", {1})
      .node("LeakyRelu", {"x"}, {"y"}, {ProtoWriter{}.bytes(1, "alpha").bytes(21, "a").varint(20, 1)})
      .output("y");
  CHECK(onnx::read_onnx(function_ref.bytes()).error().code == RefusalCode::OnnxUnsupported);
}

TEST_CASE("onnx : ModelProto sans graphe ou avec deux graphes refusé", "[gpu_ir][onnx]") {
  ProtoWriter none;
  none.varint(1, 8).message(8, ProtoWriter{}.varint(2, 17));
  CHECK(onnx::read_onnx(none.data()).error().code == RefusalCode::OnnxMalformed);

  ModelBuilder m;
  m.input("x", {1}).node("Relu", {"x"}, {"y"}).output("y");
  ProtoWriter twice = m.model();
  twice.message(7, ProtoWriter{});
  CHECK(onnx::read_onnx(twice.data()).error().code == RefusalCode::OnnxMalformed);

  CHECK(onnx::read_onnx({}).error().code == RefusalCode::OnnxMalformed);
}

TEST_CASE("onnx : champs inconnus ignorés, types filaires vérifiés", "[gpu_ir][onnx]") {
  ModelBuilder m;
  m.input("x", {2}).node("Relu", {"x"}, {"y"}).output("y");
  m.graph().varint(9999, 5).fixed64(13, 7).bytes(10, "documentation");
  ProtoWriter model = m.model();
  model.bytes(6, "doc").fixed32(20, 1);
  auto read = onnx::read_onnx(model.data());
  REQUIRE(read);
  CHECK(read->graph.nodes.size() == 1);

  // Le graphe annoncé en varint au lieu d'un message.
  ProtoWriter bad;
  bad.varint(7, 1);
  CHECK(onnx::read_onnx(bad.data()).error().code == RefusalCode::OnnxMalformed);
  // Un nom de nœud en fixed32.
  ModelBuilder name;
  name.input("x", {1}).output("y");
  name.graph().message(1, ProtoWriter{}.fixed32(3, 0).bytes(4, "Relu"));
  CHECK(onnx::read_onnx(name.bytes()).error().code == RefusalCode::OnnxMalformed);
}

// Comportement repris de animejanai-inference-rtx@118c272 src/onnx_shape.cpp
// (aji_onnx_input) : une dimension sans valeur ni nom vaut -1, et une entrée
// qui est aussi un initialiseur n'est pas une entrée réelle (vérifié dans import_test).
TEST_CASE("onnx : dimension sans valeur ni nom inconnue, comme onnx_shape.cpp", "[gpu_ir][onnx]") {
  ModelBuilder m;
  m.node("Relu", {"x"}, {"y"}).output("y");
  ProtoWriter shape;
  shape.message(1, ProtoWriter{}.varint(1, 1)).message(1, ProtoWriter{});
  ProtoWriter tensor_type;
  tensor_type.varint(1, onnx::kFloat16).message(2, shape);
  m.graph().message(11, ProtoWriter{}.bytes(1, "x").message(2, ProtoWriter{}.message(1, tensor_type)));
  auto read = onnx::read_onnx(m.bytes());
  REQUIRE(read);
  REQUIRE(read->graph.inputs.size() == 1);
  CHECK(read->graph.inputs[0].elem_type == onnx::kFloat16);
  CHECK(read->graph.inputs[0].dims == std::vector<std::int64_t>{1, -1});
  CHECK(read->graph.inputs[0].symbols == std::vector<std::string>{"", "?"});
}
