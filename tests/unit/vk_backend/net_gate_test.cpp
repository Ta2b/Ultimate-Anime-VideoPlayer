// La porte de bout en bout de l'autotuner (V9, net_gate.h) : le réseau aux choix de l'autotuner rejoué
// contre ses noyaux de référence. Des variantes justes passent la porte ; les mêmes choix n'ont aucun écart.
#include "gpu/vulkan/backend/net_gate.h"

#include <algorithm>
#include <cstddef>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "../gpu_ir/onnx_builder.h"
#include "gpu/ir/import.h"
#include "gpu/ir/passes.h"
#include "gpu/ir/shape.h"
#include "gpu/vulkan/backend/net_lower.h"
#include "net_rig.h"
#include "test_models.h"

using namespace ajp::gpu;
using namespace ajp::gpu::vk::test;

namespace {

ir::Graph build(bool half) {
  ir::Graph graph = require_value(ir::load_onnx(span_like(half).bytes()));
  require_ok(ir::infer_shapes(graph));
  (void)require_value(ir::optimize(graph));
  return graph;
}

}  // namespace

TEST_CASE("Porte de bout en bout : les noyaux de référence n'ont aucun écart, des variantes justes passent",
          "[vk_backend][net][gate]") {
  NetRig rig = make_net_rig();
  const bool half = rig.target.f16_storage;
  const ir::Graph graph = build(half);
  const auto reference = require_value(vk::net::reference_choices(graph, rig.target));
  REQUIRE(reference.size() == graph.nodes.size());
  for (std::size_t k = 0; k < graph.nodes.size(); ++k) {
    const auto options = require_value(vk::net::candidates(graph, k, rig.target));
    CHECK(std::ranges::find(options, reference[k]) != options.end());
  }
  CHECK(require_value(vk::net::network_drift(*rig.context, *rig.memory, *rig.pipelines, graph, reference,
                                             reference, rig.target)) == 0.0);
  // La dernière candidate de chaque couche (fp16 empaquetée, Winograd, autre tuile…) : juste, donc dans la
  // porte, et gardée telle quelle.
  std::vector<vk::net::KernelChoice> other;
  for (std::size_t k = 0; k < graph.nodes.size(); ++k)
    other.push_back(require_value(vk::net::candidates(graph, k, rig.target)).back());
  const double drift = require_value(
      vk::net::network_drift(*rig.context, *rig.memory, *rig.pipelines, graph, other, reference, rig.target));
  INFO("écart " << drift);
  CHECK(drift < vk::net::kNetworkDrift);
  const auto gated = require_value(
      vk::net::gate_network(*rig.context, *rig.memory, *rig.pipelines, graph, other, rig.target));
  CHECK(gated.fallback.empty());
  CHECK(gated.choices == other);
}
