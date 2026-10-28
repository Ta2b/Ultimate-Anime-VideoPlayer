// « Évaluer ma machine » (docs/ux-parametres.md, 4.7) : la décision. Le lecteur joue quelques secondes de
// vidéo avec la chaîne d'un niveau (auto_config.h) et relève ce que la présentation en a fait ; la
// recherche monte d'un niveau tant que la machine tient, redescend sinon, et garde le plus haut niveau qui
// tient. Le résultat remplace la tranche estimée de la carte (gpu_tier.h) et se garde dans bench.toml.
// Ici rien ne joue : ce sont les mesures, la règle de décision et le fichier. Le lancement des essais est
// dans auto_bench_run.h.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "app/play/gpu_tier.h"
#include "core/types.h"

namespace ajp::app::play {

// Ce qu'un essai a mesuré (les chiffres de FrameStats, frame_stats.h, sur toute la durée de l'essai).
struct BenchRun {
  Tier level = Tier::Normal;
  std::int64_t presents = 0;       // présentations de l'écran pendant l'essai
  std::int64_t dropped = 0;        // images arrivées trop tard, jamais montrées
  std::int64_t missing = 0;        // images manquées (GPU en retard), hors celles volontaires
  std::int64_t lag_p95_us = 0;     // retard de présentation, 95e centile
  std::int64_t period_us = 0;      // période de l'écran ; 0 : inconnue (60 Hz)
};

inline constexpr double kMaxLossRate = 0.02;  // images perdues ou manquées, par présentation
inline constexpr int kMaxLagPeriods = 2;      // retard p95 toléré, en périodes de l'écran
inline constexpr std::int64_t kMinPresents = 120;  // sous ce nombre, l'essai ne dit rien

// La machine tient-elle ce niveau ? Un essai trop court ne tient pas.
bool holds(const BenchRun& run);

// Le prochain niveau à essayer : l'estimé d'abord ; après un essai qui tient, le niveau au-dessus ; après un
// qui ne tient pas, celui en dessous ; fini (nullopt) quand le plus haut niveau qui tient est connu.
std::optional<Tier> next_level(const std::vector<BenchRun>& runs, Tier estimated);

// Le plus haut niveau qui tient ; Économe si aucun ne tient.
Tier best_level(const std::vector<BenchRun>& runs);

// Le résultat gardé : pour cette carte, ce niveau, à cette date.
struct BenchResult {
  std::string gpu;
  Tier level = Tier::Normal;
  std::string date;  // AAAA-MM-JJ, donnée par l'appelant
};

std::string format_bench(const BenchResult& result);
std::expected<BenchResult, Refusal> parse_bench(std::string_view toml);
std::expected<BenchResult, Refusal> load_bench(const std::filesystem::path& file);  // absent ou illisible : refus FileUnreadable
std::expected<void, Refusal> save_bench(const std::filesystem::path& file, const BenchResult& result);

// La tranche à appliquer : la mesure si elle est de cette carte, sinon l'estimation.
Tier effective_tier(const GpuFacts& gpu, const std::optional<BenchResult>& measured);

}  // namespace ajp::app::play
