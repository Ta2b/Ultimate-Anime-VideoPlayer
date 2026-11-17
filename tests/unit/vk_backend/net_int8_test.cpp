// Graphes INT8 (paires Q/DQ INT8 fusionnées, gpu/ir/onnx_fp8.h) sous le moteur de réseaux : le noyau direct
// en float arrondit à la grille INT8 comme l'interpréteur ; la variante coopmat_int8
// (net_conv_coopmat_int8.glsl) est proposée là où le pilote annonce des tuiles s8 → s32, abaissée avec ses
// entiers et ses échelles, et, sur un appareil qui les a, rend ce que rend le noyau direct sur la même entrée
// float16.
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <string>
#include <variant>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "../gpu_ir/onnx_builder.h"
#include "gpu/ir/fp8_weights.h"
#include "gpu/ir/graph.h"
#include "gpu/ir/import.h"
#include "gpu/ir/int8.h"
#include "gpu/ir/passes.h"
#include "gpu/ir/shape.h"
#include "gpu/vulkan/backend/net_lower.h"
#include "kernels/ref/half.h"
#include "net_rig.h"
#include "test_models.h"

using namespace ajp::gpu;
using namespace ajp::gpu::vk::test;
using ir::test::attr_ints;
using ir::test::ModelBuilder;

namespace {

constexpr float kInScale = 0.02F;

ir::Graph build(const ModelBuilder& model) {
  ir::Graph graph = require_value(ir::load_onnx(model.bytes()));
  require_ok(ir::infer_shapes(graph));
  (void)require_value(ir::optimize(graph));
  return graph;
}

// Des poids sur la grille INT8 d'une échelle par canal de sortie (amax / 127), en float16 si half.
std::vector<float> int8_weights(const std::vector<std::int64_t>& dims, std::size_t seed, bool half) {
  std::size_t count = 1;
  for (const std::int64_t d : dims) count *= static_cast<std::size_t>(d);
  std::vector<float> w = weights(count, seed, 0.3F);
  const std::size_t per = count / static_cast<std::size_t>(dims[0]);
  for (std::size_t c = 0; c < static_cast<std::size_t>(dims[0]); ++c) {
    float amax = 0.0F;
    for (std::size_t k = c * per; k < (c + 1) * per; ++k) amax = std::max(amax, std::abs(w[k]));
    const float s = amax / 127.0F;
    for (std::size_t k = c * per; k < (c + 1) * per; ++k) {
      w[k] = ir::round_to_int8(w[k] / s) * s;
      if (half) w[k] = ajp::kernels::ref::round_to_half(w[k]);
    }
  }
  return w;
}

// Conv 3 × 3 + Relu, puis FakeQuantInt8 et Conv 3 × 3 aux poids INT8 : la seconde lit, en Nhwc16, son entrée
// arrondie à la grille INT8 (ConvAttrs::in_quant, quant_grid Int8).
ModelBuilder int8_chain(bool half, std::int64_t channels) {
  ModelBuilder m;
  const std::int32_t type = half ? ir::onnx::kFloat16 : ir::onnx::kFloat;
  m.input("x", {1, channels, 10, 12}, type);
  add_weight(m, "w1", {channels, channels, 3, 3}, half, 81);
  add_weight(m, "b1", {channels}, half, 82);
  const std::vector<std::int64_t> w2_dims{channels, channels, 3, 3};
  const std::vector<float> w2 = int8_weights(w2_dims, 83, half);
  if (half) {
    m.initializer(ir::test::tensor_f16("w2", w2_dims, w2));
    m.initializer(ir::test::tensor_f16("s", {}, std::vector<float>{kInScale}));
  } else {
    m.initializer(ir::test::tensor_f32("w2", w2_dims, w2));
    m.initializer(ir::test::tensor_f32("s", {}, std::vector<float>{kInScale}));
  }
  add_weight(m, "b2", {channels}, half, 84);
  const auto same = attr_ints("pads", {1, 1, 1, 1});
  m.node("Conv", {"x", "w1", "b1"}, {"c1"}, {same}).node("Relu", {"c1"}, {"r1"});
  m.node("FakeQuantInt8", {"r1", "s"}, {"q1"});
  m.node("Conv", {"q1", "w2", "b2"}, {"y"}, {same}).output("y", type);
  return m;
}

// L'indice de la Conv quantifiée du graphe.
std::size_t quantized_conv(const ir::Graph& graph) {
  for (std::size_t k = 0; k < graph.nodes.size(); ++k) {
    const auto* attrs = std::get_if<ir::ConvAttrs>(&graph.nodes[k].attrs);
    if (attrs != nullptr && attrs->in_quant > 0.0F) return k;
  }
  FAIL("Conv quantifiée absente");
  return 0;
}

bool has_variant(const std::vector<vk::net::KernelChoice>& options, std::string_view variant) {
  return std::ranges::any_of(options, [&](const vk::net::KernelChoice& c) { return c.variant == variant; });
}

}  // namespace

TEST_CASE("Réseau : FakeQuantInt8 seul et Conv arrondie en INT8, noyau direct contre l'interpréteur",
          "[vk_backend][net][int8]") {
  NetRig rig = make_net_rig();
  ModelBuilder m;
  m.input("x", {1, 6, 10, 12});
  add_weight(m, "w1", {8, 6, 3, 3}, false, 71, 0.3F);
  add_weight(m, "b1", {8}, false, 72);
  add_weight(m, "w2", {8, 8, 1, 1}, false, 73, 0.3F);
  m.initializer(ir::test::tensor_f32("s1", {}, std::vector<float>{0.02F}));
  m.initializer(ir::test::tensor_f32("s2", {}, std::vector<float>{0.05F}));
  m.initializer(ir::test::tensor_f32("s3", {}, std::vector<float>{0.01F}));
  m.node("FakeQuantInt8", {"x", "s1"}, {"xq"});
  m.node("Conv", {"xq", "w1", "b1"}, {"t1"}, {attr_ints("pads", {1, 1, 1, 1})});
  m.node("FakeQuantInt8", {"t1", "s2"}, {"t1q"});
  m.node("Conv", {"t1q", "w2"}, {"t2"});
  m.node("Add", {"t2", "t1q"}, {"sum"});
  m.node("Relu", {"sum"}, {"r"});
  m.node("FakeQuantInt8", {"r", "s3"}, {"y"});  // lu par la sortie du graphe : un nœud à part
  m.output("y");
  const ir::Graph graph = build(m);
  std::size_t quantized = 0;
  for (const ir::Node& node : graph.nodes) {
    const auto* attrs = std::get_if<ir::ConvAttrs>(&node.attrs);
    if (attrs == nullptr || !(attrs->in_quant > 0.0F || attrs->out_quant > 0.0F)) continue;
    CHECK(attrs->quant_grid == ir::QuantGrid::Int8);
    ++quantized;
  }
  REQUIRE(quantized == 1);
  CHECK(std::ranges::count(graph.nodes, ir::OpKind::FakeQuantInt8, &ir::Node::op) == 1);
  const std::vector<ir::NamedValue> inputs{{"x", test_values(ir::DataType::F32, {1, 6, 10, 12}, 74)}};
  const std::vector<vk::net::KernelChoice> defaults =
      require_value(vk::net::default_choices(graph, rig.target));
  check_graph(rig, graph, inputs, kFp32, defaults);
  // Chaque variante de chaque Conv, les autres gardant la leur.
  for (std::size_t k = 0; k < graph.nodes.size(); ++k) {
    const auto options = require_value(vk::net::candidates(graph, k, rig.target));
    for (std::size_t c = 1; c < options.size(); ++c) {
      std::vector<vk::net::KernelChoice> choices = defaults;
      choices[k] = options[c];
      INFO("nœud " << k << " : " << options[c].variant);
      check_graph(rig, graph, inputs, kFp32, choices);
    }
  }
}

TEST_CASE("Réseau : variante coopmat_int8 proposée sur tuiles INT8 et abaissée avec ses entiers",
          "[vk_backend][net][int8]") {
  NetRig rig = make_net_rig();
  const ir::Graph graph = build(int8_chain(true, 16));
  const std::size_t conv = quantized_conv(graph);
  const auto& attrs = std::get<ir::ConvAttrs>(graph.nodes[conv].attrs);
  REQUIRE(attrs.quant_grid == ir::QuantGrid::Int8);
  const float in_scale = ajp::kernels::ref::round_to_half(kInScale);  // l'échelle du modèle float16
  REQUIRE(attrs.in_quant == in_scale);
  vk::net::Target target = rig.target;
  target.f16_storage = true;
  target.matrix_units = true;
  if (target.subgroup_size == 0) target.subgroup_size = 32;
  target.matrix_i8 = false;
  target.matrix_f8 = true;  // des tuiles FP8 ne servent pas une Conv de la grille INT8
  auto options = require_value(vk::net::candidates(graph, conv, target));
  CHECK_FALSE(has_variant(options, "coopmat_int8"));
  CHECK_FALSE(has_variant(options, "coopmat_fp8"));
  CHECK_FALSE(has_variant(options, "coopmat"));  // une Conv quantifiée ne passe pas par les tuiles float16
  target.matrix_i8 = true;
  options = require_value(vk::net::candidates(graph, conv, target));
  REQUIRE(has_variant(options, "coopmat_int8"));
  CHECK(options.front().variant == "direct");  // le noyau direct reste le choix par défaut

  const auto chosen =
      std::ranges::find(options, std::string("coopmat_int8"), &vk::net::KernelChoice::variant);
  std::vector<vk::net::KernelChoice> choices = require_value(vk::net::default_choices(graph, target));
  choices[conv] = *chosen;
  const vk::net::Program program = require_value(vk::net::lower(graph, choices, target));
  const auto d =
      std::ranges::find(program.dispatches, static_cast<std::uint32_t>(conv), &vk::net::Dispatch::node);
  REQUIRE(d != program.dispatches.end());
  CHECK(d->shader == "net_conv_coopmat_int8_f16");
  // IN_Q puis OUT_Q en dernier.
  CHECK(std::bit_cast<float>(d->constants[d->constants.size() - 2]) == in_scale);
  CHECK(std::bit_cast<float>(d->constants.back()) == 0.0F);
  // Poids : [ky][kx][cin][cout arrondi à 16] en entiers INT8 ; biais puis IN_Q × échelle de chaque canal.
  const ir::Tensor& w = graph.tensors[graph.nodes[conv].inputs[ir::kConvW]];
  const auto recovered = ir::recover_int8_weights(w.value);
  REQUIRE(recovered);
  const auto& wb = program.blobs[d->bindings[1].index];
  REQUIRE(wb.size() == std::size_t{9} * 16 * 16);
  for (std::size_t co = 0; co < 16; ++co) {
    for (std::size_t ci = 0; ci < 16; ++ci) {
      const std::size_t tap = 4;  // ky = kx = 1
      CHECK(static_cast<std::uint8_t>(wb[(((tap * 16) + ci) * 16) + co]) ==
            recovered->bytes[(((co * 16) + ci) * 9) + tap]);
    }
  }
  const auto& bb = program.blobs[d->bindings[2].index];
  REQUIRE(bb.size() == std::size_t{32} * sizeof(float));
  for (std::size_t co = 0; co < 16; ++co) {
    float scale = 0.0F;
    std::memcpy(&scale, bb.data() + ((16 + co) * sizeof(float)), sizeof(float));
    CHECK(scale == in_scale * recovered->scale[co]);
  }

  // Sur un appareil qui a les tuiles INT8 : la variante rend ce que rend le noyau direct sur la même entrée.
  if (!rig.target.matrix_i8) {
    WARN("pas de tuiles INT8 sur " << rig.context->caps().device_name
                                   << " : la variante coopmat_int8 n'est pas exécutée ici");
    return;
  }
  const std::vector<ir::NamedValue> inputs{{"x", test_values(ir::DataType::F16, {1, 16, 10, 12}, 85)}};
  const std::vector<vk::net::KernelChoice> defaults =
      require_value(vk::net::default_choices(graph, rig.target));
  const std::vector<ir::Value> reference = run_on_device(rig, graph, inputs, defaults);
  for (const vk::net::KernelChoice& option : require_value(vk::net::candidates(graph, conv, rig.target))) {
    if (option.variant != "coopmat_int8") continue;
    std::vector<vk::net::KernelChoice> with = defaults;
    with[conv] = option;
    const std::vector<ir::Value> got = run_on_device(rig, graph, inputs, with);
    REQUIRE(got.size() == reference.size());
    require_close(got[0], reference[0], kFp16, "coopmat_int8 contre le noyau direct");
  }
}
