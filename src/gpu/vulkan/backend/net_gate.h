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
#include <functional>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"
#include "gpu/backend.h"
#include "gpu/ir/ir.h"
#include "gpu/vulkan/backend/net_autotune.h"
#include "gpu/vulkan/backend/net_program.h"
#include "gpu/vulkan/compute/context.h"
#include "gpu/vulkan/compute/memory.h"
#include "gpu/vulkan/compute/pipeline.h"

namespace ajp::gpu::vk::net {

inline constexpr double kNetworkDrift = 0.01;
inline constexpr double kNetworkDriftLow = 0.02;
// Perte permise sous le plancher des noyaux de référence, en dB. La qualité d'image passe avant la vitesse
// (B45-01) : 0,5 dB, un bruit d'arrondi qui reste sous le seuil de ce qu'on voit, et non les 3 dB d'avant (le
// bruit d'arrondi × 1,4). Mesuré sur lavapipe (docs/moteurs.md) : les sommes float16 partout coûtent 4,8 dB à
// SPAN b5f48 sur une image réelle, 0,1 dB à RIFE 4.25 : SPAN garde donc ses sommes en float, RIFE peut
// garder la précision réduite si la porte la laisse passer.
inline constexpr double kGateLossDb = 0.5;
// Des noyaux de référence qui reproduisent le jumeau à l'octet près (graphe en float32) donnent kGateExactDb
// (200 dB) au plancher ; une variante qui ne l'est pas bit à bit (l'ordre des sommes change, découpe de K)
// tombe vers 100 dB et serait rejetée pour un écart d'arrondi invisible (C45-04). Un plancher exact vaut donc
// kGateExactFloorDb dans le verdict : très au-dessus de tout plancher d'un réseau float16 (75 à 82 dB), donc
// sans effet sur eux.
inline constexpr double kGateExactDb = 200.0;
inline constexpr double kGateExactFloorDb = 90.0;
// Au-dessus de ce PSNR contre le calcul en float, une perte ne se voit pas, quel que soit le plancher :
// l'écart quadratique moyen reste sous le quart d'un niveau sur 255. La règle des 0,5 dB seule renvoyait SPAN
// b5f48 en sommes float (70 dB contre 75) : sur une GeForce Ada, les Tensor Cores en sommes float vont deux
// fois moins vite, SPAN tourne à chaque image affichée, et l'interpolation sous Vulkan perdait près de la
// moitié de ses images par seconde face à la 0.10.3 pour une différence invisible.
inline constexpr double kGateInvisibleDb = 60.0;

// Ce que la porte fait des choix de l'autotuner, depuis les PSNR contre le jumeau en float : celui des noyaux
// de référence (le plancher), celui des choix, et celui des mêmes choix sans précision réduite s'ils en ont.
enum class GateVerdict : std::uint8_t { Keep, FullPrecision, Reference };
constexpr GateVerdict gate_verdict(double floor_db, double db, std::optional<double> full_precision_db) {
  // Un graphe en float32 (plancher exact) garde la règle stricte : la précision y est voulue.
  const bool exact = floor_db >= kGateExactDb;
  const double floor = exact ? kGateExactFloorDb : floor_db;
  const auto passes = [&](double got) {
    return got >= floor - kGateLossDb || (!exact && got >= kGateInvisibleDb);
  };
  if (passes(db)) return GateVerdict::Keep;
  if (full_precision_db && passes(*full_precision_db)) return GateVerdict::FullPrecision;
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

// Arrêt et progression de la porte (C45-02) : elle rejoue le réseau trois à quatre fois, ce qui dure quelques
// secondes sur une RTX et bien plus sous lavapipe. L'arrêt se lit entre deux rejeux ; report reçoit une
// fraction de [from, to] à chaque jeu de choix jugé.
struct GateWatch {
  std::stop_token stop;
  std::function<void(float fraction, const std::string& stage)> report;
  float from = 0.0F;
  float to = 1.0F;
};

// Le PSNR, contre le jumeau en float et sur le dessin de la porte, des sorties du réseau à chacun des jeux de
// choix donnés, dans leur ordre. Refus : ceux de l'abaissement, de l'allocation et de l'exécution.
struct Fidelity {
  std::vector<double> db;
};
std::expected<Fidelity, Refusal> network_fidelity(VkContext& context, MemoryAllocator& memory,
                                                  PipelineCache& pipelines, const ir::Graph& graph,
                                                  std::span<const std::vector<KernelChoice>> sets,
                                                  const Target& target, const GateWatch& watch = {});

struct Gated {
  std::vector<KernelChoice> choices;  // ceux qui passent la porte
  bool against_float = false;         // jugés contre le jumeau en float (db, floor_db), sinon par drift
  double db = 0.0;                    // PSNR des choix de l'autotuner contre le jumeau en float
  double floor_db = 0.0;              // celui des noyaux de référence
  double drift = 0.0;                 // porte d'avant : l'écart des choix de l'autotuner
  std::string unjudged;               // pourquoi le jumeau en float n'a pas été joué
  std::string fallback;               // vide : gardés tels quels ; sinon ce qui a été repris
};
// L'avis à donner à l'utilisateur quand des couches restent à précision réduite après la porte : leur nombre,
// la perte contre le calcul en float et le gain de temps mesuré par l'autotuner (`measured`, ses relevés ;
// « non mesuré » pour une couche reprise du cache). Vide sans couche à précision réduite.
std::string reduced_precision_notice(std::string_view model, std::span<const KernelChoice> choices,
                                     const Gated& gated, std::span<const ReducedChoice> measured);

// Le même avis, chiffré, pour l'interface (gpu::Artifact::reduced, C45-01) : le lecteur le met en phrase.
PrecisionNotice precision_notice(std::span<const KernelChoice> choices, const Gated& gated,
                                 std::span<const ReducedChoice> measured);

// Les choix de l'autotuner s'ils passent la porte ; sinon ceux qui la passent (voir plus haut).
std::expected<Gated, Refusal> gate_network(VkContext& context, MemoryAllocator& memory,
                                           PipelineCache& pipelines, const ir::Graph& graph,
                                           std::vector<KernelChoice> tuned, const Target& target,
                                           const GateWatch& watch = {});

}  // namespace ajp::gpu::vk::net
