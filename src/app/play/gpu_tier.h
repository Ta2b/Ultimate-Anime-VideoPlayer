// La tranche de performance d'une carte graphique, pour les réglages automatiques
// (docs/ux-parametres.md, 4.7). Elle vient de data/gpu_tiers.toml, embarqué dans le lecteur : des règles
// par famille (« rtx 4070 », « rx 7900 », « arc a7 »…), la première qui correspond gagne ; une carte
// inconnue reçoit une estimation par fabricant, VRAM et unités matricielles. Ce sont des estimations,
// que « Évaluer ma machine » corrige. Rien ici ne change un réglage.
#pragma once

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

#include "app/play/first_run.h"
#include "core/gpu_device.h"
#include "core/types.h"

namespace ajp::app::play {

enum class Tier : std::uint8_t { Economy, Light, Normal, Fast, Ultra };
inline constexpr int kTierCount = 5;

// « Économe », « Léger », « Normal », « Rapide », « Ultra ».
std::string_view tier_label(Tier tier);

struct TierRule {
  GpuVendor vendor = GpuVendor::Other;
  std::vector<std::string> contains;  // en minuscules, toutes requises
  Tier tier = Tier::Normal;
  bool integrated = false;
};

struct TierTable {
  std::vector<TierRule> rules;
};

// Lit un data/gpu_tiers.toml. Refus pour un TOML mal formé, un fabricant ou une tranche inconnus, une
// règle sans `contains`.
std::expected<TierTable, Refusal> parse_tier_table(std::string_view toml);
// La table embarquée (le fichier de dépôt data/gpu_tiers.toml).
const TierTable& builtin_tier_table();
// Le texte brut embarqué (data/gpu_tiers.toml).
extern const std::string_view kGpuTiersToml;

struct TierEstimate {
  Tier tier = Tier::Light;
  bool matched = false;     // une règle de la table : sinon estimation par les faits
  bool integrated = false;  // iGPU
};

// La tranche de la carte : une règle de la table, sinon une estimation par fabricant, VRAM, unités
// matricielles (cœurs Tensor, WMMA). Le nom est normalisé (minuscules, sans « (tm) » ni « (r) »).
TierEstimate estimate_tier(const TierTable& table, const GpuFacts& gpu);
inline TierEstimate estimate_tier(const GpuFacts& gpu) {
  return estimate_tier(builtin_tier_table(), gpu);
}

}  // namespace ajp::app::play
