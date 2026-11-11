// La porte de bout en bout de l'autotuner (V9, net_gate.h) : le réseau aux choix de l'autotuner rejoué
// contre son jumeau en float. Des variantes justes passent la porte ; une perte de plus de kGateLossDb sous
// les noyaux de référence ne passe pas.
#include "gpu/vulkan/backend/net_gate.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "../gpu_ir/onnx_builder.h"
#include "gpu/ir/import.h"
#include "gpu/ir/passes.h"
#include "gpu/ir/shape.h"
#include "gpu/vulkan/backend/net_autotune.h"
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
  // La dernière candidate de chaque couche à sommes float (Winograd, autre tuile…) : juste, donc dans la
  // porte, et gardée telle quelle. Les variantes à précision réduite ne passent plus que sous 0,5 dB de perte
  // (B45-01) : elles ont leur propre cas plus bas.
  std::vector<vk::net::KernelChoice> other;
  for (std::size_t k = 0; k < graph.nodes.size(); ++k) {
    const auto options = require_value(vk::net::candidates(graph, k, rig.target));
    const auto exact =
        std::ranges::find_if(options.rbegin(), options.rend(),
                             [](const vk::net::KernelChoice& c) { return !vk::net::low_precision(c); });
    other.push_back(exact != options.rend() ? *exact : options.back());
  }
  const double drift = require_value(
      vk::net::network_drift(*rig.context, *rig.memory, *rig.pipelines, graph, other, reference, rig.target));
  INFO("écart " << drift);
  CHECK(drift < vk::net::kNetworkDrift);
  const auto gated = require_value(
      vk::net::gate_network(*rig.context, *rig.memory, *rig.pipelines, graph, other, rig.target));
  CHECK(gated.fallback.empty());
  CHECK(gated.choices == other);
}

TEST_CASE("Porte de bout en bout : le verdict suit la perte sous les noyaux de référence",
          "[vk_backend][net][gate]") {
  using vk::net::gate_verdict;
  using vk::net::GateVerdict;
  // SPAN b5f48 sur lavapipe (docs/moteurs.md) : 81,6 dB aux noyaux de référence, 74,3 dB en sommes float16.
  CHECK(gate_verdict(81.6, 81.6, std::nullopt) == GateVerdict::Keep);
  CHECK(gate_verdict(81.6, 81.2, std::nullopt) == GateVerdict::Keep);
  // La qualité d'abord (B45-01) : 2,9 dB de perte, que l'ancienne porte à 3 dB gardait, ne passent plus ; la
  // précision pleine reprend la place si elle tient, sinon les noyaux de référence.
  CHECK(gate_verdict(81.6, 78.7, std::nullopt) == GateVerdict::Reference);
  CHECK(gate_verdict(81.6, 78.7, 81.4) == GateVerdict::FullPrecision);
  CHECK(gate_verdict(81.6, 80.9, 81.4) == GateVerdict::FullPrecision);
  CHECK(vk::net::kGateLossDb <= 0.5);
  CHECK(gate_verdict(81.6, 74.3, 81.2) == GateVerdict::FullPrecision);
  CHECK(gate_verdict(81.6, 74.3, 70.0) == GateVerdict::Reference);
  CHECK(gate_verdict(81.6, 74.3, std::nullopt) == GateVerdict::Reference);
  // RIFE 4.25 : 77,9 contre 77,8 dB, gardé.
  CHECK(gate_verdict(77.9, 77.8, 77.9) == GateVerdict::Keep);
  // Une valeur non finie dans la sortie : jamais gardée.
  const double broken = -std::numeric_limits<double>::infinity();
  CHECK(gate_verdict(81.6, broken, 81.6) == GateVerdict::FullPrecision);
  CHECK(gate_verdict(81.6, broken, broken) == GateVerdict::Reference);
}

TEST_CASE("Porte de bout en bout : le jumeau en float et la fidélité des noyaux de référence",
          "[vk_backend][net][gate]") {
  NetRig rig = make_net_rig();
  require_f16(rig);
  const ir::Graph graph = build(true);
  const ir::Graph twin = vk::net::float_twin(graph);
  REQUIRE(twin.tensors.size() == graph.tensors.size());
  for (const ir::Tensor& t : twin.tensors) {
    CHECK(t.type != ir::DataType::F16);
    CHECK(t.value.type != ir::DataType::F16);
  }
  const auto reference = require_value(vk::net::reference_choices(graph, rig.target));
  std::vector<vk::net::KernelChoice> low = reference;
  bool any_low = false;
  for (std::size_t k = 0; k < graph.nodes.size(); ++k) {
    const auto options = require_value(vk::net::candidates(graph, k, rig.target));
    const auto found = std::ranges::find_if(options, vk::net::low_precision);
    if (found == options.end()) continue;
    low[k] = *found;
    any_low = true;
  }
  const std::vector<std::vector<vk::net::KernelChoice>> sets{reference, low};
  const auto fidelity = require_value(
      vk::net::network_fidelity(*rig.context, *rig.memory, *rig.pipelines, graph, sets, rig.target));
  REQUIRE(fidelity.db.size() == 2);
  INFO("référence " << fidelity.db[0] << " dB, précision réduite " << fidelity.db[1] << " dB");
  // Rangement en float16, sommes en float : loin au-dessus de ce qui se voit, mais pas exact.
  CHECK(fidelity.db[0] > 55.0);
  CHECK(fidelity.db[0] < 200.0);
  CHECK(fidelity.db[1] > 40.0);
  // Les choix sans précision réduite reviennent aux noyaux de référence ou aux sommes en float.
  const auto wider = vk::net::full_precision(low, reference);
  CHECK(std::ranges::none_of(wider, vk::net::low_precision));
  if (!any_low) CHECK(wider == reference);
  // La porte entière : ce qu'elle garde ne perd pas plus que la marge sous le plancher.
  const auto gated =
      require_value(vk::net::gate_network(*rig.context, *rig.memory, *rig.pipelines, graph, low, rig.target));
  if (any_low) {
    CHECK(gated.against_float);
    const std::vector<std::vector<vk::net::KernelChoice>> kept{gated.choices};
    const auto after = require_value(
        vk::net::network_fidelity(*rig.context, *rig.memory, *rig.pipelines, graph, kept, rig.target));
    CHECK(after.db[0] >= fidelity.db[0] - vk::net::kGateLossDb);
  }
}

TEST_CASE("Porte de bout en bout : une précision réduite gardée s'annonce avec sa perte et son gain",
          "[vk_backend][net][gate]") {
  using vk::net::KernelChoice;
  using vk::net::ReducedChoice;
  const std::vector<KernelChoice> choices{{.variant = "direct"}, {.variant = "coopmat_acc16"}};
  vk::net::Gated gated{.choices = choices, .against_float = true, .db = 80.9, .floor_db = 81.6};
  // Aucune couche à précision réduite : rien à dire.
  CHECK(vk::net::reduced_precision_notice("m", std::vector<KernelChoice>{{.variant = "direct"}}, gated, {})
            .empty());
  // Gain mesuré : 100 µs en précision réduite contre 150 µs en sommes float, soit +50 %.
  const std::vector<ReducedChoice> measured{
      {.node = 1, .variant = "coopmat_acc16", .chosen_us = 100.0, .full_precision_us = 150.0}};
  const std::string notice = vk::net::reduced_precision_notice("span", choices, gated, measured);
  CHECK(notice.contains("précision réduite"));
  CHECK(notice.contains("+50 %"));
  CHECK(notice.contains("-0.70 dB"));
  // Choix repris du cache : le gain n'est pas mesuré, et l'avis le dit.
  const std::vector<ReducedChoice> cached{{.node = 1, .variant = "coopmat_acc16"}};
  CHECK(vk::net::reduced_precision_notice("span", choices, gated, cached).contains("non mesuré"));
  // Un relevé d'une autre variante au même nœud (reprise par la porte) ne chiffre rien.
  const std::vector<ReducedChoice> stale{
      {.node = 1, .variant = "direct_f16p", .chosen_us = 10.0, .full_precision_us = 100.0}};
  CHECK(vk::net::reduced_precision_notice("span", choices, gated, stale).contains("non mesuré"));
  // Porte jugée sans le jumeau en float : la perte n'est pas chiffrée, elle est dite non mesurée.
  gated.against_float = false;
  CHECK(vk::net::reduced_precision_notice("span", choices, gated, measured).contains("perte non mesurée"));
}
