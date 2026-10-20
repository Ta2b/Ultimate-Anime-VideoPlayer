// Convolutions du moteur de réseaux contre l'interpréteur CPU FP32 : chaque variante candidate
// (directe float, fp16 empaquetée, Winograd, matrices coopératives là où l'appareil les autorise), avec
// groupes, pas, dilatation, marges, épilogues fondus, puis petits graphes façon SPAN et RIFE de
// bout en bout. FP32 : 1e-4 relatif. FP16 : chaque tenseur intermédiaire est arrondi à 2^-11
// près par le GPU et pas par l'interpréteur, d'où kFp16 (net_rig.h).
#include <algorithm>
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
using ir::test::tensor_f16;
using ir::test::tensor_f32;

namespace {

ir::Graph build(const ModelBuilder& model) {
  ir::Graph graph = require_value(ir::load_onnx(model.bytes()));
  require_ok(ir::infer_shapes(graph));
  (void)require_value(ir::optimize(graph));
  return graph;
}

std::vector<float> weights(std::size_t count, std::size_t seed, float scale = 0.3F) {
  return test_values(ir::DataType::F32, {static_cast<std::int64_t>(count)}, seed, -scale, scale).f;
}

// Poids et biais d'une Conv ou ConvTranspose, en float ou en float16 selon le modèle.
void add_weight(ModelBuilder& m, const std::string& name, const std::vector<std::int64_t>& dims, bool half,
                std::size_t seed, float scale = 0.3F) {
  std::size_t count = 1;
  for (const std::int64_t d : dims) count *= static_cast<std::size_t>(d);
  const std::vector<float> values = weights(count, seed, scale);
  if (half) {
    m.initializer(tensor_f16(name, dims, values));
  } else {
    m.initializer(tensor_f32(name, dims, values));
  }
}

// Chaque nœud de convolution essaie chacune de ses variantes, les autres gardant la leur.
void check_all_variants(NetRig& rig, const ir::Graph& graph, std::span<const ir::NamedValue> inputs,
                        Tolerance tolerance) {
  const std::vector<vk::net::KernelChoice> defaults =
      require_value(vk::net::default_choices(graph, rig.target));
  check_graph(rig, graph, inputs, tolerance, defaults);
  for (std::size_t k = 0; k < graph.nodes.size(); ++k) {
    const auto options = require_value(vk::net::candidates(graph, k, rig.target));
    for (std::size_t c = 1; c < options.size(); ++c) {
      std::vector<vk::net::KernelChoice> choices = defaults;
      choices[k] = options[c];
      std::string tuning;
      for (const std::uint32_t t : options[c].tuning) tuning += " " + std::to_string(t);
      INFO("nœud " << k << " (" << graph.nodes[k].name << ") : " << options[c].variant << tuning);
      check_graph(rig, graph, inputs, tolerance, choices);
    }
  }
}

struct ConvCase {
  std::int64_t n = 1;
  std::int64_t cin = 5;
  std::int64_t cout = 7;
  std::int64_t h = 9;
  std::int64_t w = 11;
  std::vector<std::int64_t> kernel{3, 3};
  std::vector<ir::test::ProtoWriter> attrs;
  bool bias = true;
  std::int64_t group = 1;
};

ModelBuilder conv_model(const ConvCase& c, bool half) {
  ModelBuilder m;
  const std::int32_t type = half ? ir::onnx::kFloat16 : ir::onnx::kFloat;
  m.input("x", {c.n, c.cin, c.h, c.w}, type);
  add_weight(m, "w", {c.cout, c.cin / c.group, c.kernel[0], c.kernel[1]}, half, 11);
  if (c.bias) add_weight(m, "b", {c.cout}, half, 12);
  std::vector<ir::test::ProtoWriter> attrs = c.attrs;
  attrs.push_back(attr_int("group", c.group));
  ir::test::ProtoWriter node;
  node.bytes(1, "x").bytes(1, "w");
  if (c.bias) node.bytes(1, "b");
  node.bytes(2, "y").bytes(3, "conv").bytes(4, "Conv");
  for (const auto& a : attrs) node.message(5, a);
  m.graph().message(1, node);
  m.output("y", type);
  return m;
}

// Conv 3x3 + Relu → Conv 3x3 + résiduel → Conv 1x1 + Sigmoid → LeakyRelu : activations
// internes en Nhwc16, épilogues fondus par les passes.
ModelBuilder chain_model(bool half, std::int64_t cin, std::int64_t mid) {
  ModelBuilder m;
  const std::int32_t type = half ? ir::onnx::kFloat16 : ir::onnx::kFloat;
  m.input("x", {1, cin, 13, 17}, type);
  add_weight(m, "w1", {mid, cin, 3, 3}, half, 21);
  add_weight(m, "b1", {mid}, half, 22);
  add_weight(m, "w2", {mid, mid, 3, 3}, half, 23, 0.1F);
  add_weight(m, "b2", {mid}, half, 24);
  add_weight(m, "w3", {8, mid, 1, 1}, half, 25);
  add_weight(m, "b3", {8}, half, 26);
  const auto same = attr_ints("pads", {1, 1, 1, 1});
  m.node("Conv", {"x", "w1", "b1"}, {"c1"}, {same}).node("Relu", {"c1"}, {"r1"});
  m.node("Conv", {"r1", "w2", "b2"}, {"c2"}, {same}).node("Add", {"c2", "r1"}, {"s2"});
  m.node("Conv", {"s2", "w3", "b3"}, {"c3"}).node("Sigmoid", {"c3"}, {"g3"});
  m.node("LeakyRelu", {"g3"}, {"y"}, {attr_float("alpha", 0.2F)}).output("y", type);
  return m;
}

}  // namespace

TEST_CASE("Réseau : Conv directe, toutes géométries, toutes variantes", "[vk_backend][net][conv]") {
  NetRig rig = make_net_rig();
  const std::vector<ConvCase> cases{
      {.attrs = {attr_ints("pads", {1, 1, 1, 1})}},
      {.n = 2,
       .cin = 6,
       .cout = 8,
       .h = 12,
       .w = 10,
       .attrs = {attr_ints("strides", {2, 1}), attr_ints("dilations", {2, 2}),
                 attr_ints("pads", {2, 1, 1, 0})},
       .group = 2},
      {.cin = 12, .cout = 20, .kernel = {1, 1}, .bias = false},
      {.cin = 4,
       .cout = 9,
       .h = 7,
       .w = 9,
       .kernel = {4, 4},
       .attrs = {attr_ints("strides", {2, 2}), attr_string("auto_pad", "SAME_UPPER")}},
      {.cin = 6, .cout = 6, .h = 5, .w = 6, .attrs = {attr_ints("pads", {1, 1, 1, 1})}, .group = 6},
  };
  for (std::size_t k = 0; k < cases.size(); ++k) {
    INFO("cas " << k);
    const ir::Graph graph = build(conv_model(cases[k], false));
    const ConvCase& c = cases[k];
    const std::vector<ir::NamedValue> inputs{
        {"x", test_values(ir::DataType::F32, {c.n, c.cin, c.h, c.w}, k)}};
    check_all_variants(rig, graph, inputs, kFp32);
  }
}

TEST_CASE("Réseau : chaîne de Conv en Nhwc16 avec épilogues fondus", "[vk_backend][net][conv]") {
  NetRig rig = make_net_rig();
  const ir::Graph graph = build(chain_model(false, 5, 16));
  CHECK(std::ranges::count(graph.tensors, ir::Layout::Nhwc16, &ir::Tensor::layout) >= 3);
  const std::vector<ir::NamedValue> inputs{{"x", test_values(ir::DataType::F32, {1, 5, 13, 17}, 31)}};
  check_all_variants(rig, graph, inputs, kFp32);
}

// mid impair (13) : le dernier canal de Winograd, lu seul.
TEST_CASE("Réseau : Conv en float16, variantes fp16, Winograd et matrices coopératives",
          "[vk_backend][net][conv]") {
  NetRig rig = make_net_rig();
  require_f16(rig);
  for (const auto& [cin, mid] : {std::pair<std::int64_t, std::int64_t>{5, 16}, {12, 48}, {16, 20}, {7, 13}}) {
    INFO("cin " << cin << ", mid " << mid);
    const ir::Graph graph = build(chain_model(true, cin, mid));
    const std::vector<ir::NamedValue> inputs{{"x", test_values(ir::DataType::F16, {1, cin, 13, 17}, 32)}};
    check_all_variants(rig, graph, inputs, kFp16);
  }
  if (!rig.target.matrix_units) {
    WARN("matrices coopératives non autorisées sur " << rig.context->caps().device_name
                                                     << " : seule leur non-sélection est vérifiée");
  }
}

TEST_CASE("Réseau : ConvTranspose", "[vk_backend][net][conv]") {
  NetRig rig = make_net_rig();
  struct Case {
    std::int64_t cin;
    std::int64_t cout;
    std::int64_t group;
    std::vector<ir::test::ProtoWriter> attrs;
  };
  const std::vector<Case> cases{
      {6, 4, 1, {attr_ints("strides", {2, 2}), attr_ints("pads", {1, 1, 1, 1})}},
      {6,
       4,
       2,
       {attr_ints("strides", {2, 2}), attr_ints("pads", {1, 1, 1, 1}), attr_ints("output_padding", {1, 0})}},
      {3, 5, 1, {attr_ints("strides", {1, 2}), attr_ints("dilations", {2, 1})}},
  };
  for (std::size_t k = 0; k < cases.size(); ++k) {
    const Case& c = cases[k];
    ModelBuilder m;
    m.input("x", {2, c.cin, 5, 6});
    add_weight(m, "w", {c.cin, c.cout / c.group, 4, 4}, false, 40 + k);
    add_weight(m, "b", {c.cout}, false, 50 + k);
    std::vector<ir::test::ProtoWriter> attrs = c.attrs;
    attrs.push_back(attr_int("group", c.group));
    ir::test::ProtoWriter node;
    node.bytes(1, "x").bytes(1, "w").bytes(1, "b").bytes(2, "y").bytes(3, "deconv").bytes(4, "ConvTranspose");
    for (const auto& a : attrs) node.message(5, a);
    m.graph().message(1, node);
    m.output("y");
    INFO("cas " << k);
    const ir::Graph graph = build(m);
    const std::vector<ir::NamedValue> inputs{{"x", test_values(ir::DataType::F32, {2, c.cin, 5, 6}, k)}};
    check_all_variants(rig, graph, inputs, kFp32);
  }
}

namespace {

// SPAN en petit : conv 3x3 + PRelu, deux fois, conv 3x3 vers 12 canaux, pixel shuffle x2.
ModelBuilder span_like(bool half) {
  ModelBuilder m;
  const std::int32_t type = half ? ir::onnx::kFloat16 : ir::onnx::kFloat;
  m.input("x", {1, 3, 24, 32}, type);
  add_weight(m, "w1", {16, 3, 3, 3}, half, 61);
  add_weight(m, "b1", {16}, half, 62);
  add_weight(m, "p1", {16, 1, 1}, half, 63);
  add_weight(m, "w2", {16, 16, 3, 3}, half, 64, 0.1F);
  add_weight(m, "b2", {16}, half, 65);
  add_weight(m, "p2", {16, 1, 1}, half, 66);
  add_weight(m, "w3", {12, 16, 3, 3}, half, 67, 0.1F);
  add_weight(m, "b3", {12}, half, 68);
  const auto same = attr_ints("pads", {1, 1, 1, 1});
  m.node("Conv", {"x", "w1", "b1"}, {"c1"}, {same}).node("PRelu", {"c1", "p1"}, {"a1"});
  m.node("Conv", {"a1", "w2", "b2"}, {"c2"}, {same}).node("PRelu", {"c2", "p2"}, {"a2"});
  m.node("Conv", {"a2", "w3", "b3"}, {"c3"}, {same});
  m.node("DepthToSpace", {"c3"}, {"y"}, {attr_int("blocksize", 2), attr_string("mode", "CRD")});
  m.output("y", type);
  return m;
}

// RIFE en petit : deux images, réduction linéaire, conv, ConvTranspose vers un flot, grille par
// Transpose, GridSample bord et coins alignés en float, mélange par un masque sigmoïde.
ModelBuilder rife_like() {
  ModelBuilder m(16);
  m.input("x", {1, 6, 16, 24});
  m.ints("split", {3, 3});
  m.weight("scales", {4}, {1.0F, 1.0F, 0.5F, 0.5F});
  add_weight(m, "w1", {8, 6, 3, 3}, false, 71);
  add_weight(m, "b1", {8}, false, 72);
  add_weight(m, "wt", {8, 3, 4, 4}, false, 73, 0.2F);
  add_weight(m, "bt", {3}, false, 74);
  m.ints("flow_shape", {1, 1, 16, 24});
  m.node("Split", {"x", "split"}, {"a", "b"}, {attr_int("axis", 1)});
  m.node("Concat", {"a", "b"}, {"ab"}, {attr_int("axis", 1)});
  m.node("Resize", {"ab", "", "scales"}, {"small"}, {attr_string("mode", "linear")});
  m.node("Conv", {"small", "w1", "b1"}, {"c1"}, {attr_ints("pads", {1, 1, 1, 1})});
  m.node("LeakyRelu", {"c1"}, {"l1"}, {attr_float("alpha", 0.2F)});
  m.node("ConvTranspose", {"l1", "wt", "bt"}, {"up"},
         {attr_ints("strides", {2, 2}), attr_ints("pads", {1, 1, 1, 1})});
  m.ints("starts", {0}).ints("ends", {2}).ints("axes", {1});
  m.node("Slice", {"up", "starts", "ends", "axes"}, {"flow"});
  m.ints("mask_start", {2}).ints("mask_end", {3});
  m.node("Slice", {"up", "mask_start", "mask_end", "axes"}, {"logit"});
  m.node("Transpose", {"flow"}, {"grid"}, {attr_ints("perm", {0, 2, 3, 1})});
  m.node(
      "GridSample", {"a", "grid"}, {"warped"},
      {attr_string("mode", "bilinear"), attr_string("padding_mode", "border"), attr_int("align_corners", 1)});
  m.node("Sigmoid", {"logit"}, {"mask"});
  m.node("Mul", {"warped", "mask"}, {"wa"});
  m.node("Sub", {"b", "wa"}, {"y"});
  m.output("y");
  return m;
}

}  // namespace

TEST_CASE("Réseau : graphe façon SPAN de bout en bout", "[vk_backend][net][graph]") {
  NetRig rig = make_net_rig();
  {
    const ir::Graph graph = build(span_like(false));
    const std::vector<ir::NamedValue> inputs{{"x", test_values(ir::DataType::F32, {1, 3, 24, 32}, 81)}};
    check_graph(rig, graph, inputs, kFp32);
  }
  if (!rig.target.f16_storage) SKIP("pas de stockage 16 bits : SPAN en float16 non vérifié");
  const ir::Graph graph = build(span_like(true));
  const std::vector<ir::NamedValue> inputs{{"x", test_values(ir::DataType::F16, {1, 3, 24, 32}, 82)}};
  check_all_variants(rig, graph, inputs, kFp16);
}

TEST_CASE("Réseau : graphe façon RIFE de bout en bout", "[vk_backend][net][graph]") {
  NetRig rig = make_net_rig();
  const ir::Graph graph = build(rife_like());
  const std::vector<ir::NamedValue> inputs{{"x", test_values(ir::DataType::F32, {1, 6, 16, 24}, 91)}};
  check_all_variants(rig, graph, inputs, kFp32);
}
