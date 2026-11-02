// La porte de bout en bout de l'autotuner (V9) : chaque couche choisie concorde avec le noyau direct à
// 3 % près (net_autotune.h), mais des écarts permis s'additionnent d'une couche à l'autre. Le réseau
// entier, aux choix de l'autotuner, est donc rejoué une fois contre le même réseau aux noyaux de
// référence (le direct de chaque couche) sur les mêmes entrées. Trop loin (écart quadratique au-delà de
// kNetworkDrift de l'énergie de la référence, ou une valeur non finie) : les sommes en float16 cèdent
// d'abord leur place aux sommes en float, puis, si l'écart reste, tout le réseau reprend les noyaux de
// référence.
#pragma once

#include <expected>
#include <span>
#include <string>
#include <vector>

#include "core/types.h"
#include "gpu/ir/ir.h"
#include "gpu/vulkan/backend/net_program.h"
#include "gpu/vulkan/compute/context.h"
#include "gpu/vulkan/compute/memory.h"
#include "gpu/vulkan/compute/pipeline.h"

namespace ajp::gpu::vk::net {

inline constexpr double kNetworkDrift = 0.10;

// Le noyau de référence de chaque nœud : la variante « direct » quand le nœud la propose, sinon la
// première de ses candidates. Refus : ceux de candidates().
std::expected<std::vector<KernelChoice>, Refusal> reference_choices(const ir::Graph& graph,
                                                                    const Target& target);

// L'écart entre les sorties du réseau aux choix `tuned` et aux choix `reference`, mêmes entrées (motif de
// tuning_input ramené à [0, 1]) : la racine de Σ(a − b)² / Σ b² sur toutes les sorties ; +∞ si une
// valeur n'est pas finie. Refus : ceux de l'abaissement, de l'allocation et de l'exécution.
std::expected<double, Refusal> network_drift(VkContext& context, MemoryAllocator& memory,
                                             PipelineCache& pipelines, const ir::Graph& graph,
                                             std::span<const KernelChoice> tuned,
                                             std::span<const KernelChoice> reference, const Target& target);

struct Gated {
  std::vector<KernelChoice> choices;  // ceux qui passent la porte
  double drift = 0.0;                 // l'écart des choix de l'autotuner
  std::string fallback;               // vide : gardés tels quels ; sinon ce qui a été repris
};
// Les choix de l'autotuner s'ils passent la porte ; sinon ceux qui la passent (voir plus haut).
std::expected<Gated, Refusal> gate_network(VkContext& context, MemoryAllocator& memory,
                                           PipelineCache& pipelines, const ir::Graph& graph,
                                           std::vector<KernelChoice> tuned, const Target& target);

}  // namespace ajp::gpu::vk::net
