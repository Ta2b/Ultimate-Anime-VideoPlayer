// Import ONNX vers l'IR, puis inférence des formes.
#include <cstdint>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/ir/ops.h"
#include "test_util.h"

using ajp::Refusal;
using ajp::RefusalCode;
using namespace ajp::gpu::ir;
using namespace ajp::gpu::ir::test;

namespace {

Refusal load_refusal(const ModelBuilder& m) {
  auto graph = load_onnx(m.bytes());
  REQUIRE_FALSE(graph);
  return graph.error();
}

const Tensor& tensor_named(const Graph& graph, std::string_view name) {
  const TensorId id = find_tensor(graph, name).value_or(kNoTensor);
  REQUIRE(id != kNoTensor);
  return graph.tensors[id];
}

}  // namespace

TEST_CASE("import : un opérateur inconnu est refusé avec son nom", "[gpu_ir][import]") {
  ModelBuilder m;
  m.input("x", {1, 4}).node("Relu", {"x"}, {"r"}).node("Einsum", {"r"}, {"y"}).output("y");
  const Refusal refused = load_refusal(m);
  CHECK(refused.code == RefusalCode::OperatorUnknown);
  CHECK(refused.detail == "Einsum");

  ModelBuilder custom;
  custom.input("x", {1, 4}).node("FusedConv", {"x"}, {"y"}, {}, "com.microsoft").output("y");
  const Refusal domain = load_refusal(custom);
  CHECK(domain.code == RefusalCode::OperatorUnknown);
  CHECK(domain.detail == "com.microsoft:FusedConv");

  // Un nom d'opérateur de l'IR dans un autre domaine reste inconnu.
  ModelBuilder relu_elsewhere;
  relu_elsewhere.input("x", {1}).node("Relu", {"x"}, {"y"}, {}, "ai.onnx.ml").output("y");
  CHECK(load_refusal(relu_elsewhere).code == RefusalCode::OperatorUnknown);
}

TEST_CASE("import : chaque opérateur du jeu de l'IR est reconnu", "[gpu_ir][import]") {
  for (const std::string_view op :
       {"Conv",      "ConvTranspose", "Add",       "Sub",          "Mul",          "Div",    "Relu",
        "LeakyRelu", "PRelu",         "Sigmoid",   "Tanh",         "Clip",         "Concat", "Slice",
        "Split",     "Reshape",       "Transpose", "DepthToSpace", "SpaceToDepth", "Resize", "GridSample",
        "Pad",       "Cast",          "Identity",  "ReduceMean",   "AveragePool"}) {
    INFO(op);
    CHECK(op_from_onnx(op));
  }
  CHECK_FALSE(op_from_onnx("Constant"));  // devenu tenseur constant à l'import
  CHECK_FALSE(op_from_onnx("MatMul"));
}

TEST_CASE("import : opsets hors de la plage connue refusés", "[gpu_ir][import]") {
  for (const std::int64_t opset : {kMinOpset - 1, kMaxOpset + 1}) {
    ModelBuilder m(opset);
    m.input("x", {1}).node("Relu", {"x"}, {"y"}).output("y");
    CHECK(load_refusal(m).code == RefusalCode::OnnxUnsupported);
  }
  ModelBuilder none;
  none.input("x", {1}).node("Relu", {"x"}, {"y"}).output("y");
  ProtoWriter model;
  model.message(7, none.graph()).message(8, ProtoWriter{}.bytes(1, "com.microsoft").varint(2, 1));
  CHECK(load_onnx(model.data()).error().code == RefusalCode::OnnxUnsupported);  // pas d'opset ONNX
}

TEST_CASE("import : nœuds Constant devenus tenseurs constants", "[gpu_ir][import]") {
  ModelBuilder m;
  m.input("x", {2})
      .node("Constant", {}, {"k"}, {attr_tensor("value", tensor_f32("", {2}, std::vector<float>{3, 4}))})
      .node("Constant", {}, {"s"}, {attr_float("value_float", 0.5F)})
      .node("Constant", {}, {"axes"}, {attr_ints("value_ints", {0})})
      .node("Mul", {"x", "k"}, {"m"})
      .node("Mul", {"m", "s"}, {"y"})
      .output("y");
  const Graph graph = build(m);
  CHECK(graph.nodes.size() == 2);
  CHECK(tensor_named(graph, "k").kind == TensorKind::Constant);
  CHECK(tensor_named(graph, "k").value.f == std::vector<float>{3, 4});
  CHECK(tensor_named(graph, "s").value.dims.empty());
  CHECK(tensor_named(graph, "axes").value.i == std::vector<std::int64_t>{0});
  CHECK(execute1(graph, {{.name = "x", .value = floats({2}, {1, 2})}}).f == std::vector<float>{1.5F, 4});

  ModelBuilder sparse;
  sparse.input("x", {1}).node("Constant", {}, {"k"}, {attr_string("value_string", "a")}).output("k");
  CHECK(load_refusal(sparse).code == RefusalCode::OnnxUnsupported);
}

// Comportement repris de animejanai-inference-rtx@118c272 src/onnx_shape.cpp
// (aji_onnx_input) : une entrée qui figure aussi parmi les initialiseurs est un poids.
TEST_CASE("import : une entrée doublée d'un initialiseur est un poids, comme onnx_shape.cpp",
          "[gpu_ir][import]") {
  ModelBuilder m;
  m.input("w", {2}).input("x", {2}).weight("w", {2}, {5, 6}).node("Add", {"x", "w"}, {"y"}).output("y");
  const Graph graph = build(m);
  REQUIRE(graph.inputs.size() == 1);
  CHECK(graph.tensors[graph.inputs[0]].name == "x");
  CHECK(tensor_named(graph, "w").kind == TensorKind::Constant);
}

TEST_CASE("import : ordre des nœuds rétabli, cycles et références absentes refusés", "[gpu_ir][import]") {
  ModelBuilder reversed;
  reversed.input("x", {3}).node("Relu", {"a"}, {"y"}).node("Sigmoid", {"x"}, {"a"}).output("y");
  const Graph graph = build(reversed);
  REQUIRE(graph.nodes.size() == 2);
  CHECK(graph.nodes[0].op == OpKind::Sigmoid);
  CHECK(graph.nodes[1].op == OpKind::Relu);

  ModelBuilder cycle;
  cycle.input("x", {3}).node("Add", {"x", "b"}, {"a"}).node("Relu", {"a"}, {"b"}).output("b");
  CHECK(load_refusal(cycle).code == RefusalCode::GraphInvalid);

  ModelBuilder missing;
  missing.input("x", {3}).node("Add", {"x", "ghost"}, {"y"}).output("y");
  CHECK(load_refusal(missing).code == RefusalCode::GraphInvalid);

  ModelBuilder twice;
  twice.input("x", {3}).node("Relu", {"x"}, {"y"}).node("Tanh", {"x"}, {"y"}).output("y");
  CHECK(load_refusal(twice).code == RefusalCode::GraphInvalid);

  ModelBuilder no_output;
  no_output.input("x", {3}).node("Relu", {"x"}, {"y"});
  CHECK(load_refusal(no_output).code == RefusalCode::GraphInvalid);

  ModelBuilder arity;
  arity.input("x", {3}).node("Relu", {"x", "x"}, {"y"}).output("y");
  CHECK(load_refusal(arity).code == RefusalCode::GraphInvalid);
}

TEST_CASE("import : attributs d'anciens opsets passés en entrée", "[gpu_ir][import]") {
  ModelBuilder unsqueeze(11);
  unsqueeze.input("x", {3}).node("Unsqueeze", {"x"}, {"y"}, {attr_ints("axes", {0, 2})}).output("y");
  const Graph graph = build(unsqueeze);
  REQUIRE(graph.nodes[0].inputs.size() == 2);
  CHECK(graph.tensors[graph.nodes[0].inputs[1]].value.i == std::vector<std::int64_t>{0, 2});
  CHECK(graph.tensors[graph.nodes[0].outputs[0]].dims == Dims{1, 3, 1});

  // Dans un opset récent, axes est une entrée : l'attribut est refusé.
  ModelBuilder modern(13);
  modern.input("x", {3}).node("Unsqueeze", {"x"}, {"y"}, {attr_ints("axes", {0})}).output("y");
  CHECK(load_refusal(modern).code == RefusalCode::GraphInvalid);
}

TEST_CASE("formes : entrée symbolique à donner, puis vérifiée", "[gpu_ir][shape]") {
  ModelBuilder m;
  m.input("x", {1, 3, -1, -1}).node("Relu", {"x"}, {"y"}).output("y");
  CHECK(refusal(m).code == RefusalCode::ShapeInvalid);
  CHECK(refusal(m, {{.name = "x", .dims = {1, 4, 8, 8}}}).code ==
        RefusalCode::ShapeInvalid);                                                         // canal déclaré 3
  CHECK(refusal(m, {{.name = "x", .dims = {1, 3, 8}}}).code == RefusalCode::ShapeInvalid);  // rang
  CHECK(refusal(m, {{.name = "z", .dims = {1, 3, 8, 8}}}).code == RefusalCode::GraphInvalid);  // nom
  CHECK(refusal(m, {{.name = "x", .dims = {1, 3, 1 << 25, 1}}}).code == RefusalCode::ShapeInvalid);
  CHECK(refusal(m, {{.name = "x", .dims = {1, 3, 1 << 20, 1 << 20}}}).code == RefusalCode::ShapeInvalid);
  const Graph graph = build(m, {{.name = "x", .dims = {1, 3, 8, 6}}});
  CHECK(tensor_named(graph, "y").dims == Dims{1, 3, 8, 6});
  CHECK(tensor_named(graph, "y").known);
}

TEST_CASE("formes : Reshape et Resize dont les paramètres dérivent de Shape", "[gpu_ir][shape]") {
  // Taille de sortie calculée dans le graphe : Shape -> Slice -> Mul -> Concat -> Resize.
  ModelBuilder m;
  m.input("x", {1, 3, -1, -1})
      .ints("s", {2})
      .ints("e", {4})
      .ints("two", {2})
      .ints("nc", {1, 3})
      .ints("flat", {0, 0, -1})
      .node("Shape", {"x"}, {"shape"})
      .node("Slice", {"shape", "s", "e"}, {"hw"})
      .node("Mul", {"hw", "two"}, {"hw2"})
      .node("Concat", {"nc", "hw2"}, {"sizes"}, {attr_int("axis", 0)})
      .node("Resize", {"x", "", "", "sizes"}, {"big"})
      .node("Reshape", {"big", "flat"}, {"y"})
      .output("y");
  const Graph graph = build(m, {{.name = "x", .dims = {1, 3, 5, 7}}});
  CHECK(tensor_named(graph, "sizes").dims == Dims{4});
  CHECK(tensor_named(graph, "big").dims == Dims{1, 3, 10, 14});
  CHECK(tensor_named(graph, "y").dims == Dims{1, 3, 140});
  // Les valeurs de forme restent des activations : seul le pliage les rend constantes.
  CHECK(tensor_named(graph, "sizes").kind == TensorKind::Activation);
  const Value y = execute1(graph, {{.name = "x", .value = random_floats({1, 3, 5, 7})}});
  CHECK(y.dims == Dims{1, 3, 140});
}

TEST_CASE("formes : une forme impossible est refusée au nœud fautif", "[gpu_ir][shape]") {
  ModelBuilder m;
  m.input("x", {1, 3, 4, 4}).weight("w", {8, 2, 3, 3}, std::vector<float>(144, 0.0F));
  m.node("Conv", {"x", "w"}, {"y"}).output("y");
  const Refusal refused = refusal(m);
  CHECK(refused.code == RefusalCode::ShapeInvalid);
  CHECK(refused.detail.find("Conv") != std::string::npos);
}
