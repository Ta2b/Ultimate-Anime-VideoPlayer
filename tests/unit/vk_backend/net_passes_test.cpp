// Les passes du moteur Vulkan (engine_passes) gardent les sorties du réseau : l'interpréteur sur le graphe
// importé contre l'interpréteur sur le graphe que le moteur abaisse, sur RIFE et SPAN en petit (A44-11).
// L'entrée est une image lisse décalée d'une image à l'autre : sur du bruit, un réseau à flot optique
// amplifie le moindre arrondi et la concordance ne mesure plus les passes.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "../gpu_ir/test_util.h"
#include "gpu/ir/graph.h"
#include "gpu/ir/onnx_model.h"
#include "gpu/ir/passes.h"
#include "gpu/vulkan/backend/net_build.h"
#include "kernels/ref/half.h"
#include "net_rig.h"
#include "test_models.h"

using namespace ajp;
using namespace ajp::gpu;
using namespace ajp::gpu::vk::test;

namespace {

// Dégradés sinusoïdaux dans [0,1 ; 0,9], un par canal, décalés de deux pixels toutes les trois couches.
ir::Value smooth_input(const ir::Tensor& tensor) {
  ir::Value value = ir::make_value(tensor.type, tensor.dims);
  const auto width = static_cast<std::size_t>(tensor.dims[3]);
  const std::size_t plane = width * static_cast<std::size_t>(tensor.dims[2]);
  for (std::size_t k = 0; k < value.f.size(); ++k) {
    const std::size_t c = k / plane;
    const auto x = static_cast<float>((k % width) + (c >= 3 ? 2U : 0U));
    const auto y = static_cast<float>((k % plane) / width);
    const float v = 0.5F + (0.4F * std::sin((0.21F * x) + (0.17F * y) + static_cast<float>(c % 3)));
    value.f[k] = tensor.type == ir::DataType::F16 ? kernels::ref::round_to_half(v) : v;
  }
  return value;
}

double psnr_db(const std::vector<float>& got, const std::vector<float>& want) {
  double error = 0.0;
  for (std::size_t k = 0; k < want.size(); ++k) {
    const double d = static_cast<double>(got[k]) - static_cast<double>(want[k]);
    error += d * d;
  }
  if (error == 0.0) return 200.0;
  return 10.0 * std::log10(static_cast<double>(want.size()) / error);
}

// La concordance, en dB, entre le graphe importé et celui que le moteur abaisse pour cette cible.
double passes_fidelity(const ir::test::ModelBuilder& model, bool half, bool matrices) {
  const ir::Graph before = ir::test::build(model);
  ir::Graph after = before;
  const vk::net::Target target{.f16_storage = true, .matrix_units = matrices};
  require_ok(vk::net::engine_passes(after, "essai", half, target));
  require_ok(ir::validate(after));
  std::vector<ir::NamedValue> inputs;
  for (const ir::TensorId id : before.inputs)
    inputs.push_back({.name = before.tensors[id].name, .value = smooth_input(before.tensors[id])});
  const std::vector<ir::Value> want = run_reference(before, inputs);
  const std::vector<ir::Value> got = run_reference(after, inputs);
  REQUIRE(got.size() == want.size());
  REQUIRE(got[0].dims == want[0].dims);
  return psnr_db(got[0].f, want[0].f);
}

// Le bloc résiduel de RIFE : Conv, Mul par une constante par canal, Add de l'entrée, LeakyRelu, deux fois.
// Le Mul se fond dans les poids de la Conv (fuse_conv_bias), arrondis en float16 dans un modèle float16.
ir::test::ModelBuilder scaled_blocks(bool half) {
  ir::test::ModelBuilder m(17);
  const std::int32_t type = half ? ir::onnx::kFloat16 : ir::onnx::kFloat;
  m.input("x", {1, 8, 12, 16}, type);
  std::string from = "x";
  for (std::size_t block = 0; block < 2; ++block) {
    const std::string tag = std::to_string(block);
    add_weight(m, "w" + tag, {8, 8, 3, 3}, half, 81 + (3 * block), 0.15F);
    add_weight(m, "b" + tag, {8}, half, 82 + (3 * block), 0.1F);
    add_weight(m, "beta" + tag, {1, 8, 1, 1}, half, 83 + (3 * block), 0.4F);
    m.node("Conv", {from, "w" + tag, "b" + tag}, {"c" + tag}, {ir::test::attr_ints("pads", {1, 1, 1, 1})});
    m.node("Mul", {"c" + tag, "beta" + tag}, {"s" + tag});
    m.node("Add", {"s" + tag, from}, {"r" + tag});
    m.node("LeakyRelu", {"r" + tag}, {block == 1 ? "y" : "l" + tag}, {ir::test::attr_float("alpha", 0.2F)});
    from = "l" + tag;
  }
  m.output("y", type);
  return m;
}

}  // namespace

TEST_CASE("Réseau : les passes du moteur gardent la sortie de RIFE et de SPAN en float",
          "[vk_backend][net][passes]") {
  for (const bool matrices : {false, true}) {
    CAPTURE(matrices);
    // En float, seules les réécritures jouent : l'écart est celui de l'ordre des sommes.
    CHECK(passes_fidelity(rife_like(false), false, matrices) > 100.0);
    CHECK(passes_fidelity(span_like(false), false, matrices) > 100.0);
    CHECK(passes_fidelity(scaled_blocks(false), false, matrices) > 100.0);
  }
}

TEST_CASE("Réseau : les passes du moteur gardent la sortie de RIFE et de SPAN en float16",
          "[vk_backend][net][passes]") {
  for (const bool matrices : {false, true}) {
    CAPTURE(matrices);
    // En float16, les poids fondus ou réécrits sont arrondis une fois (pass_fuse.cpp) : au-dessus du pas
    // d'une image 8 bits (59 dB), loin d'un défaut de passe (moins de 30 dB).
    CHECK(passes_fidelity(rife_like(true), true, matrices) > 60.0);
    CHECK(passes_fidelity(span_like(true), true, matrices) > 60.0);
    const double scaled = passes_fidelity(scaled_blocks(true), true, matrices);
    CHECK(scaled > 60.0);
    CHECK(scaled < 200.0);  // les poids fondus sont bien arrondis : la passe a joué
  }
}

TEST_CASE("Réseau : les passes du moteur réécrivent la ConvTranspose pour les matrices coopératives",
          "[vk_backend][net][passes]") {
  ir::Graph graph = ir::test::build(rife_like(false));
  require_ok(vk::net::engine_passes(graph, "essai", false, {.f16_storage = true, .matrix_units = true}));
  CHECK(ir::test::count_ops(graph, ir::OpKind::ConvTranspose) == 0);
  ir::Graph plain = ir::test::build(rife_like(false));
  require_ok(vk::net::engine_passes(plain, "essai", false, {.f16_storage = true, .matrix_units = false}));
  CHECK(ir::test::count_ops(plain, ir::OpKind::ConvTranspose) == 1);
}
