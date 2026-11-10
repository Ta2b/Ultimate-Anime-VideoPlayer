// La porte de bout en bout de l'autotuner (V9) : chaque couche choisie concorde avec le noyau direct à sa
// tolérance (1 % en sommes float, 3 % à précision réduite, net_autotune.cpp), mais des écarts permis
// s'additionnent d'une couche à l'autre. Le réseau entier est donc rejoué une fois, sur un dessin (aplats
// sombres, traits fins, deux images décalées et leur grille pour un réseau d'interpolation), contre son
// jumeau calculé tout en float (A45-02) : la référence est le calcul exact, pas un autre réseau en float16.
// Le plancher est le PSNR des noyaux de référence (le direct de chaque couche, sommes en float, rangement en
// float16) contre ce jumeau ; les choix de l'autotuner passent s'ils perdent au plus kGateLossDb sous lui.
// Sinon les couches à précision réduite cèdent d'abord leur place (sommes en float pour coopmat_acc16, noyau
// de référence sinon), puis tout le réseau reprend les noyaux de référence.
// Quand le jumeau en float ne peut pas être joué (mémoire, opérateur sans noyau float), la porte d'avant
// juge : écart quadratique aux noyaux de référence sur du bruit, kNetworkDrift (kNetworkDriftLow si une
// couche est à précision réduite).
#pragma once

#include <cstdint>
#include <expected>
#include <optional>
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

inline constexpr double kNetworkDrift = 0.01;
inline constexpr double kNetworkDriftLow = 0.02;
// Perte permise sous le plancher des noyaux de référence, en dB : le bruit d'arrondi au plus doublé. Mesuré
// sur lavapipe (docs/moteurs.md) : les sommes float16 partout coûtent 4,8 dB à SPAN b5f48 sur une image
// réelle, 0,1 dB à RIFE 4.25.
inline constexpr double kGateLossDb = 3.0;

// Ce que la porte fait des choix de l'autotuner, depuis les PSNR contre le jumeau en float : celui des noyaux
// de référence (le plancher), celui des choix, et celui des mêmes choix sans précision réduite s'ils en ont.
enum class GateVerdict : std::uint8_t { Keep, FullPrecision, Reference };
constexpr GateVerdict gate_verdict(double floor_db, double db, std::optional<double> full_precision_db) {
  if (db >= floor_db - kGateLossDb) return GateVerdict::Keep;
  if (full_precision_db && *full_precision_db >= floor_db - kGateLossDb) return GateVerdict::FullPrecision;
  return GateVerdict::Reference;
}

// Le même graphe, tout tenseur float16 passé en float : mêmes poids, tout le calcul en float.
ir::Graph float_twin(const ir::Graph& graph);

// Les choix sans précision réduite : coopmat_acc16 reprend la même tuile en sommes float (toujours proposée à
// côté), les autres variantes à précision réduite le noyau de référence de leur couche.
std::vector<KernelChoice> full_precision(std::span<const KernelChoice> choices,
                                         std::span<const KernelChoice> reference);

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

// Le PSNR, contre le jumeau en float et sur le dessin de la porte, des sorties du réseau à chacun des jeux de
// choix donnés, dans leur ordre. Refus : ceux de l'abaissement, de l'allocation et de l'exécution.
struct Fidelity {
  std::vector<double> db;
};
std::expected<Fidelity, Refusal> network_fidelity(VkContext& context, MemoryAllocator& memory,
                                                  PipelineCache& pipelines, const ir::Graph& graph,
                                                  std::span<const std::vector<KernelChoice>> sets,
                                                  const Target& target);

struct Gated {
  std::vector<KernelChoice> choices;  // ceux qui passent la porte
  bool against_float = false;         // jugés contre le jumeau en float (db, floor_db), sinon par drift
  double db = 0.0;                    // PSNR des choix de l'autotuner contre le jumeau en float
  double floor_db = 0.0;              // celui des noyaux de référence
  double drift = 0.0;                 // porte d'avant : l'écart des choix de l'autotuner
  std::string unjudged;               // pourquoi le jumeau en float n'a pas été joué
  std::string fallback;               // vide : gardés tels quels ; sinon ce qui a été repris
};
// Les choix de l'autotuner s'ils passent la porte ; sinon ceux qui la passent (voir plus haut).
std::expected<Gated, Refusal> gate_network(VkContext& context, MemoryAllocator& memory,
                                           PipelineCache& pipelines, const ir::Graph& graph,
                                           std::vector<KernelChoice> tuned, const Target& target);

}  // namespace ajp::gpu::vk::net
