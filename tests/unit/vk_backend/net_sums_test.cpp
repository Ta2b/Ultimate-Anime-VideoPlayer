// Mesure sur demande (C05, A45-01) : ce que coûtent les variantes à précision réduite sur les vrais réseaux,
// en PSNR contre le calcul en float, sur le dessin de la porte de bout en bout (net_gate.h).
//   ajp_vk_backend_tests "[.sommes]"
// Sur lavapipe : direct_f16p et winograd_f16p. Sur une RTX : coopmat_acc16 aussi, dont l'arrondi interne des
// Tensor Cores ne se mesure que là.
#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <format>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/ir/graph.h"
#include "gpu/vulkan/backend/net_autotune.h"
#include "gpu/vulkan/backend/net_build.h"
#include "gpu/vulkan/backend/net_gate.h"
#include "gpu/vulkan/backend/net_lower.h"
#include "net_rig.h"

using namespace ajp::gpu;
using namespace ajp::gpu::vk::test;

namespace {

constexpr std::array<std::string_view, 3> kLow{"coopmat_acc16", "direct_f16p", "winograd_f16p"};

void measure(NetRig& rig, const std::string& name, const std::filesystem::path& onnx, const ir::Dims& shape) {
  if (!std::filesystem::exists(onnx)) {
    std::cout << name << " : modèle absent\n";
    return;
  }
  const vk::net::BuildRequest request{
      .onnx = onnx, .model_id = name, .model_sha256 = "", .input_shape = shape, .fp16 = true};
  const ir::Graph graph = require_value(vk::net::prepare_graph(request, rig.target));
  const auto reference = require_value(vk::net::reference_choices(graph, rig.target));
  std::vector<std::vector<vk::net::KernelChoice>> sets{reference};
  std::vector<std::string> labels{"noyaux de référence"};
  for (const std::string_view variant : kLow) {
    std::vector<vk::net::KernelChoice> low = reference;
    std::size_t layers = 0;
    for (std::size_t k = 0; k < graph.nodes.size(); ++k) {
      const auto options = require_value(vk::net::candidates(graph, k, rig.target));
      const auto found = std::ranges::find(options, variant, &vk::net::KernelChoice::variant);
      if (found == options.end()) continue;
      low[k] = *found;
      ++layers;
    }
    if (layers == 0) continue;
    sets.push_back(std::move(low));
    labels.push_back(std::format("{} sur {} couches", variant, layers));
  }
  const auto fidelity = require_value(
      vk::net::network_fidelity(*rig.context, *rig.memory, *rig.pipelines, graph, sets, rig.target));
  for (std::size_t k = 0; k < labels.size(); ++k) {
    std::cout << std::format("{} {} : {} : {:.2f} dB ({:+.2f})\n", name, ir::format_dims(shape), labels[k],
                             fidelity.db[k], fidelity.db[k] - fidelity.db[0]);
  }
  CHECK(fidelity.db[0] > 55.0);
}

}  // namespace

TEST_CASE("Mesure : précision réduite contre le calcul en float sur les vrais réseaux", "[.sommes]") {
  NetRig rig = make_net_rig();
  require_f16(rig);
  const std::filesystem::path rife(AJP_RIFE_MODEL_DIR);
  measure(rig, "SPAN b5f48", AJP_SPAN_MODEL, {1, 3, 360, 640});
  measure(rig, "RIFE 4.25", rife / "rife_v4.25.onnx", {1, 11, 384, 640});
  measure(rig, "RIFE 4.22 lite", rife / "rife_v4.22_lite.onnx", {1, 11, 384, 640});
}
