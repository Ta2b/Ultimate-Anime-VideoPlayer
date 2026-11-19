// La cadence haute (chantier 0.3 du plan : 165 i/s sur un écran 165 Hz) : ce qui bride le mode Rapide quand
// le GPU suit. Le banc du 19 novembre (RTX 4070 Ti, QD-OLED 165 Hz VRR, RIFE 4.25 seul) plafonnait vers 107
// i/s avec 3,8 ms de GPU par image créée : le calcul n'était pas la limite.
//
// - Pertes à l'écran contre pertes du GPU. Une image sautée à la présentation alors qu'elle était déjà écrite
// à
//   un tour d'avant n'est pas en retard : main l'a vue prête, puis une plus récente est devenue due avant
//   qu'il la montre (tour de boucle trop long, acquisition qui attend l'écran, gigue de l'horloge à la
//   fréquence même de l'écran). Elle élargissait l'écart du mode Rapide (×1,12 par image, jusqu'à ×2), comme
//   une image que le GPU n'avait pas livrée : à 165 Hz, quelques pertes d'affichage par seconde suffisaient à
//   tenir la contre-pression haute et la cadence vers 110 i/s. Elles sont désormais comptées à part et
//   laissées hors de la contre-pression (--display-loss-pressure : l'ancien compte).
// - Le mode de présentation de la swapchain (--swap-mode) : FIFO reste le défaut ; mailbox, immediate et
//   relaxed (FIFO relâché) sont des essais à comparer en lecture. libplacebo retombe sur FIFO si la surface
//   ne l'offre pas, et le dit au journal.
// - La marge sous le maximum d'un écran VRR (--vrr-margin, 1 % par défaut comme avant).
// - Une ligne de fin de séance dit ce qui a borné la cadence : l'écran, le temps de service ou la
//   contre-pression.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "core/free_rate.h"

namespace kirari::app::play {

class LiveSettings;
struct FrameStats;

enum class SwapMode : std::uint8_t { Fifo, Mailbox, Immediate, Relaxed };

constexpr std::optional<SwapMode> parse_swap_mode(std::string_view text) {
  if (text == "fifo") return SwapMode::Fifo;
  if (text == "mailbox") return SwapMode::Mailbox;
  if (text == "immediate") return SwapMode::Immediate;
  if (text == "relaxed") return SwapMode::Relaxed;
  return std::nullopt;
}

constexpr std::string_view name(SwapMode mode) {
  switch (mode) {
    case SwapMode::Fifo:
      return "fifo";
    case SwapMode::Mailbox:
      return "mailbox";
    case SwapMode::Immediate:
      return "immediate";
    case SwapMode::Relaxed:
      return "relaxed";
  }
  return "fifo";
}

// Le premier tour de main où chaque image de l'anneau a été vue écrite, pour les dernières images reçues.
class ReadySeen {
 public:
  void note(std::uint32_t slot, std::uint64_t value, std::int64_t tick_us);
  // Vrai si l'image était déjà vue écrite à un tour d'avant tick_us.
  bool before(std::uint32_t slot, std::uint64_t value, std::int64_t tick_us) const;

 private:
  struct Entry {
    std::uint32_t slot = 0;
    std::uint64_t value = 0;
    std::int64_t tick_us = 0;  // 0 : case libre
  };
  std::array<Entry, 32> entries_{};
  std::size_t next_ = 0;
};

// Les compteurs de FrameStats que la cadence lit.
struct LossCounts {
  std::int64_t missing = 0;        // trous dans les pts
  std::int64_t dropped = 0;        // sautées à la présentation
  std::int64_t dropped_ready = 0;  // dont prêtes à un tour d'avant : perdues par l'affichage
};

// Les pertes qui élargissent l'écart du mode Rapide, entre deux relevés (deltas). Une image sautée laisse
// aussi un trou dans les pts : celles de l'affichage s'ôtent des deux comptes, sauf si display_counts.
constexpr std::int64_t pressure_losses(const LossCounts& delta, bool display_counts) {
  const std::int64_t display = display_counts ? 0 : delta.dropped_ready;
  const std::int64_t missing = delta.missing - display;
  const std::int64_t dropped = delta.dropped - display;
  return std::max<std::int64_t>({missing, dropped, 0});
}

struct CadenceOptions {
  SwapMode swap_mode = SwapMode::Fifo;
  bool display_loss_pressure = false;  // --display-loss-pressure : l'ancien compte
  double vrr_margin_pct = 1.0;         // --vrr-margin
};

class CadenceWatch {
 public:
  explicit CadenceWatch(const CadenceOptions& options) : options_(options) {}

  // Les pertes du mode Rapide depuis le relevé précédent, à partir des totaux de FrameStats.
  std::int64_t take_losses(const LossCounts& totals);
  // Un relevé du mode Rapide (active : le mode tourne), au rythme du sondage de l'écran.
  void sample(bool active, const core::FreeInfo& info, std::int64_t min_gap_us);
  // Le relevé de main : les pertes à donner au mode Rapide (LiveSettings::free_missed), puis sa cadence.
  void poll(LiveSettings& live, const FrameStats& shown, std::int64_t now_us);
  // La ligne de fin de séance.
  std::string summary() const;

 private:
  CadenceOptions options_;
  LossCounts seen_;
  std::int64_t gpu_losses_ = 0;
  std::int64_t display_losses_ = 0;
  int samples_ = 0;
  int screen_bound_ = 0;    // l'écart planifié est l'écart minimal : l'écran borne
  int service_bound_ = 0;   // le temps de service × 1,10 dépasse l'écart minimal : le GPU borne
  int pressure_bound_ = 0;  // seule la contre-pression le dépasse
  double fps_sum_ = 0.0;
  double step_sum_ = 0.0;
  double service_sum_ = 0.0;
  double pressure_sum_ = 0.0;
  double pressure_max_ = 1.0;
  std::int64_t min_gap_us_ = 0;
};

}  // namespace kirari::app::play
