// Chaque opération sans réglage du moteur de réseaux contre l'interpréteur CPU FP32 de gpu_ir :
// élément par élément (diffusion comprise), copies à pas, pixels, échantillonnage, réductions.
// Chaque cas tourne seul (tenseurs Nchw) puis entouré de LeakyRelu, ce qui range ses tenseurs
// internes en Nhwc16 quand l'opération le permet.
#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "../gpu_ir/onnx_builder.h"
#include "gpu/ir/graph.h"
#include "gpu/ir/import.h"
#include "gpu/ir/passes.h"
#include "gpu/ir/shape.h"
#include "gpu/vulkan/backend/net_lower.h"
#include "net_rig.h"

using namespace ajp::gpu;
using namespace ajp::gpu::vk::test;
using ir::test::attr_float;
using ir::test::attr_int;
using ir::test::attr_ints;
using ir::test::attr_string;
using ir::test::ModelBuilder;
using ir::test::tensor_f32;
using ir::test::tensor_i64;

namespace {

ir::Graph build(const ModelBuilder& model, bool passes = true) {
  ir::Graph graph = require_value(ir::load_onnx(model.bytes()));
  require_ok(ir::infer_shapes(graph));
  if (passes) (void)require_value(ir::optimize(graph));
  return graph;
}

std::size_t nhwc_count(const ir::Graph& graph) {
  return static_cast<std::size_t>(std::ranges::count(graph.tensors, ir::Layout::Nhwc16, &ir::Tensor::layout));
}

// Le modèle à une opération op(x, constantes...) -> y, puis la même entourée de LeakyRelu.
struct OpCase {
  std::string op;
  std::vector<std::int64_t> x;
  std::vector<std::string> extra_inputs;  // noms d'initialiseurs déjà posés par setup
  std::vector<ir::test::ProtoWriter> attrs;
  std::function<void(ModelBuilder&)> setup;
};

ModelBuilder single(const OpCase& c, bool wrapped) {
  ModelBuilder m;
  m.input("x", c.x);
  if (c.setup) c.setup(m);
  std::vector<std::string_view> inputs{wrapped ? "wrap_in" : "x"};
  for (const std::string& name : c.extra_inputs) inputs.emplace_back(name);
  if (wrapped) m.node("LeakyRelu", {"x"}, {"wrap_in"}, {attr_float("alpha", 0.5F)});
  ir::test::ProtoWriter node;
  for (const std::string_view in : inputs) node.bytes(1, in);
  node.bytes(2, wrapped ? "wrap_out" : "y");
  node.bytes(3, c.op + "_case");
  node.bytes(4, c.op);
  for (const auto& a : c.attrs) node.message(5, a);
  m.graph().message(1, node);
  if (wrapped) m.node("LeakyRelu", {"wrap_out"}, {"y"}, {attr_float("alpha", 0.5F)});
  m.output("y");
  return m;
}

void check_op(const OpCase& c, ir::DataType type = ir::DataType::F32) {
  NetRig rig = make_net_rig();
  for (const bool wrapped : {false, true}) {
    const ir::Graph graph = build(single(c, wrapped));
    INFO(c.op << (wrapped ? " entouré" : " seul") << ", tenseurs Nhwc16 : " << nhwc_count(graph));
    const std::vector<ir::NamedValue> inputs{{"x", test_values(type, c.x, 1)}};
    check_graph(rig, graph, inputs, kFp32);
  }
}

}  // namespace

TEST_CASE("Réseau : opérations élément par élément et diffusion", "[vk_backend][net]") {
  check_op({.op = "Relu", .x = {1, 3, 7, 9}});
  check_op({.op = "Sigmoid", .x = {1, 4, 5, 6}});
  check_op({.op = "Tanh", .x = {2, 3, 4, 5}});
  check_op({.op = "LeakyRelu", .x = {1, 3, 5, 5}, .attrs = {}});
  check_op({.op = "Clip", .x = {1, 3, 6, 6}, .extra_inputs = {"lo", "hi"}, .setup = [](ModelBuilder& m) {
              m.initializer(tensor_f32("lo", {}, std::vector<float>{-0.25F}));
              m.initializer(tensor_f32("hi", {}, std::vector<float>{0.5F}));
            }});
  check_op({.op = "Add", .x = {1, 3, 8, 8}, .extra_inputs = {"b"}, .setup = [](ModelBuilder& m) {
              m.weight("b", {1, 3, 1, 1}, {0.5F, -1.0F, 2.0F});
            }});
  check_op({.op = "Mul", .x = {2, 3, 4, 5}, .extra_inputs = {"s"}, .setup = [](ModelBuilder& m) {
              m.weight("s", {5}, {1.0F, -2.0F, 0.5F, 3.0F, -0.25F});
            }});
  check_op({.op = "Sub", .x = {1, 2, 3, 4}, .extra_inputs = {"s"}, .setup = [](ModelBuilder& m) {
              m.weight("s", {}, {0.75F});
            }});
  check_op({.op = "Div", .x = {1, 2, 3, 4}, .extra_inputs = {"d"}, .setup = [](ModelBuilder& m) {
              m.weight("d", {1, 2, 1, 1}, {4.0F, -0.5F});
            }});
  check_op({.op = "PRelu", .x = {1, 3, 4, 4}, .extra_inputs = {"p"}, .setup = [](ModelBuilder& m) {
              m.weight("p", {3, 1, 1}, {0.1F, 0.2F, -0.3F});
            }});
}

TEST_CASE("Réseau : deux entrées de même forme", "[vk_backend][net]") {
  NetRig rig = make_net_rig();
  ModelBuilder m;
  m.input("x", {1, 4, 6, 5}).input("z", {1, 4, 6, 5});
  m.node("Mul", {"x", "z"}, {"p"}).node("Sub", {"p", "x"}, {"q"}).node("Add", {"q", "z"}, {"y"}).output("y");
  const ir::Graph graph = build(m);
  const std::vector<ir::NamedValue> inputs{{"x", test_values(ir::DataType::F32, {1, 4, 6, 5}, 2)},
                                           {"z", test_values(ir::DataType::F32, {1, 4, 6, 5}, 3)}};
  check_graph(rig, graph, inputs, kFp32);
}

TEST_CASE("Réseau : copies à pas (Concat, Split, Slice, Transpose, Reshape)", "[vk_backend][net]") {
  check_op({.op = "Concat",
            .x = {1, 3, 5, 6},
            .extra_inputs = {"c"},
            .attrs = {attr_int("axis", 1)},
            .setup = [](ModelBuilder& m) { m.weight("c", {1, 2, 5, 6}, std::vector<float>(60, 0.25F)); }});
  check_op({.op = "Concat",
            .x = {2, 3, 4, 5},
            .extra_inputs = {"c"},
            .attrs = {attr_int("axis", -1)},
            .setup = [](ModelBuilder& m) { m.weight("c", {2, 3, 4, 2}, std::vector<float>(48, -1.5F)); }});
  check_op({.op = "Slice",
            .x = {1, 4, 9, 10},
            .extra_inputs = {"starts", "ends", "axes", "steps"},
            .setup = [](ModelBuilder& m) {
              m.ints("starts", {1, 8, 1})
                  .ints("ends", {4, 0, 9})
                  .ints("axes", {1, 3, 2})
                  .ints("steps", {1, -2, 3});
            }});
  check_op({.op = "Transpose", .x = {2, 3, 4, 5}, .attrs = {attr_ints("perm", {0, 2, 3, 1})}});
  check_op({.op = "Reshape", .x = {2, 4, 3, 5}, .extra_inputs = {"shape"}, .setup = [](ModelBuilder& m) {
              m.ints("shape", {1, 8, 3, 5});
            }});

  NetRig rig = make_net_rig();
  ModelBuilder m;
  m.input("x", {1, 7, 4, 6}).ints("split", {2, 4, 1});
  m.node("Split", {"x", "split"}, {"a", "b", "c"}, {attr_int("axis", 1)}).output("a").output("b").output("c");
  const ir::Graph graph = build(m);
  const std::vector<ir::NamedValue> inputs{{"x", test_values(ir::DataType::F32, {1, 7, 4, 6}, 4)}};
  check_graph(rig, graph, inputs, kFp32);
}

TEST_CASE("Réseau : pixels (DepthToSpace, SpaceToDepth, Pad)", "[vk_backend][net]") {
  check_op({.op = "DepthToSpace",
            .x = {1, 8, 3, 4},
            .attrs = {attr_int("blocksize", 2), attr_string("mode", "DCR")}});
  check_op({.op = "DepthToSpace",
            .x = {2, 12, 3, 4},
            .attrs = {attr_int("blocksize", 2), attr_string("mode", "CRD")}});
  check_op({.op = "SpaceToDepth", .x = {1, 3, 6, 8}, .attrs = {attr_int("blocksize", 2)}});
  for (const char* mode : {"constant", "reflect", "edge"}) {
    check_op({.op = "Pad",
              .x = {1, 3, 6, 7},
              .extra_inputs = {"pads", "value"},
              .attrs = {attr_string("mode", mode)},
              .setup = [](ModelBuilder& m) {
                m.ints("pads", {0, 0, 2, 1, 0, 1, 3, -1});
                m.initializer(tensor_f32("value", {}, std::vector<float>{0.5F}));
              }});
  }
}

TEST_CASE("Réseau : SpaceToDepth CRD (pixel_unshuffle)", "[vk_backend][net]") {
  NetRig rig = make_net_rig();
  ModelBuilder m;
  m.input("x", {1, 3, 6, 8}).node("SpaceToDepth", {"x"}, {"y"}, {attr_int("blocksize", 2)}).output("y");
  ir::Graph graph = build(m, false);
  std::get<ir::BlockAttrs>(graph.nodes[0].attrs).mode = ir::BlockMode::Crd;
  const std::vector<ir::NamedValue> inputs{{"x", test_values(ir::DataType::F32, {1, 3, 6, 8}, 5)}};
  check_graph(rig, graph, inputs, kFp32);
}

TEST_CASE("Réseau : échantillonnage (Resize, GridSample)", "[vk_backend][net]") {
  const auto resize = [](const char* mode, const char* coord, const char* nearest,
                         std::vector<float> scales) {
    return OpCase{.op = "Resize",
                  .x = {1, 3, 6, 10},
                  .extra_inputs = {"", "scales"},
                  .attrs = {attr_string("mode", mode), attr_string("coordinate_transformation_mode", coord),
                            attr_string("nearest_mode", nearest)},
                  .setup = [scales](ModelBuilder& m) { m.weight("scales", {4}, scales); }};
  };
  check_op(resize("nearest", "asymmetric", "floor", {1.0F, 1.0F, 2.0F, 2.0F}));
  check_op(resize("nearest", "half_pixel", "round_prefer_floor", {1.0F, 1.0F, 1.5F, 0.5F}));
  check_op(resize("linear", "half_pixel", "round_prefer_floor", {1.0F, 1.0F, 0.5F, 0.5F}));
  check_op(resize("linear", "align_corners", "round_prefer_floor", {1.0F, 1.0F, 2.0F, 1.5F}));
  check_op(resize("linear", "pytorch_half_pixel", "round_prefer_floor", {1.0F, 1.0F, 0.25F, 0.5F}));

  NetRig rig = make_net_rig();
  for (const char* padding : {"border", "zeros"}) {
    for (const std::int64_t align : {0, 1}) {
      for (const char* mode : {"bilinear", "nearest"}) {
        ModelBuilder m(16);
        m.input("x", {2, 3, 7, 9}).input("g", {2, 5, 6, 2});
        m.node("GridSample", {"x", "g"}, {"y"},
               {attr_string("mode", mode), attr_string("padding_mode", padding),
                attr_int("align_corners", align)});
        m.output("y");
        const ir::Graph graph = build(m);
        INFO(mode << " " << padding << " align " << align);
        const std::vector<ir::NamedValue> inputs{
            {"x", test_values(ir::DataType::F32, {2, 3, 7, 9}, 6)},
            {"g", test_values(ir::DataType::F32, {2, 5, 6, 2}, 7, -1.2F, 1.2F)}};
        check_graph(rig, graph, inputs, kFp32);
      }
    }
  }
}

TEST_CASE("Réseau : réductions (ReduceMean, AveragePool)", "[vk_backend][net]") {
  check_op(
      {.op = "ReduceMean", .x = {1, 3, 6, 5}, .attrs = {attr_ints("axes", {2, 3}), attr_int("keepdims", 1)}});
  check_op(
      {.op = "ReduceMean", .x = {2, 4, 3, 5}, .attrs = {attr_ints("axes", {1}), attr_int("keepdims", 0)}});
  check_op({.op = "AveragePool",
            .x = {1, 3, 8, 6},
            .attrs = {attr_ints("kernel_shape", {2, 2}), attr_ints("strides", {2, 2})}});
  for (const std::int64_t include : {0, 1}) {
    check_op({.op = "AveragePool",
              .x = {1, 2, 7, 5},
              .attrs = {attr_ints("kernel_shape", {3, 3}), attr_ints("pads", {1, 1, 1, 1}),
                        attr_int("count_include_pad", include)}});
  }
}

TEST_CASE("Réseau : Cast vers float16 et calcul en float16", "[vk_backend][net]") {
  NetRig rig = make_net_rig();
  require_f16(rig);
  ModelBuilder m;
  m.input("x", {1, 3, 5, 7});
  m.node("Cast", {"x"}, {"h"}, {attr_int("to", ir::onnx::kFloat16)});
  m.node("Sigmoid", {"h"}, {"s"});
  m.node("Cast", {"s"}, {"y"}, {attr_int("to", ir::onnx::kFloat)});
  m.output("y");
  const ir::Graph graph = build(m);
  const std::vector<ir::NamedValue> inputs{
      {"x", test_values(ir::DataType::F32, {1, 3, 5, 7}, 8, -4.0F, 4.0F)}};
  check_graph(rig, graph, inputs, kFp16);
}

TEST_CASE("Réseau : opérateur sans noyau refusé à l'abaissement", "[vk_backend][net]") {
  ModelBuilder m;
  m.input("x", {1, 3, 4, 4}).node("Shape", {"x"}, {"y"}).output("y", ir::onnx::kInt64);
  const ir::Graph graph = build(m, false);
  const auto refused = vk::net::candidates(graph, 0, {});
  require_refusal(refused, ajp::RefusalCode::OperatorUnknown);
}
