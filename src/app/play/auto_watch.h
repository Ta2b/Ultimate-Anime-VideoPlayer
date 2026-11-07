// Le suivi du palier de l'Auto pendant la lecture (P5.15, sorti de run_chain) : quand le dire au journal,
// quand l'annoncer à l'écran, quand le retenir pour la prochaine lecture, quand signaler un plancher figé.
// Pur : il reçoit l'état d'un tour et rend ce qu'il faut faire ; run_chain journalise, annonce et écrit.
#pragma once

#include <cstdint>
#include <optional>

#include "core/auto_rate.h"

namespace ajp::app::play {

// L'Auto figé au plancher : dit au journal après une tentative de sortie (core::AutoRate::kFloorEscapeUs),
// puis au plus toutes les 30 s. Un palier tenu 3 s s'annonce à l'écran, tenu 6 s il s'écrit dans la mémoire
// de l'Auto (un palier qui oscille n'y entre pas). Le mode Rapide : une ligne de journal toutes les 5 s.
inline constexpr std::int64_t kFloorLogAfterUs = core::AutoRate::kFloorEscapeUs;
inline constexpr std::int64_t kFloorLogEveryUs = 30'000'000;
inline constexpr std::int64_t kAutoNotifyAfterUs = 3'000'000;
inline constexpr std::int64_t kAutoStoreAfterUs = 6'000'000;
inline constexpr std::int64_t kFreeLogEveryUs = 5'000'000;

struct AutoTurn {
  std::int64_t at_us = 0;
  int centi = 0;           // le palier en centièmes de multiplicateur ; 0 : l'Auto est coupé
  bool free = false;       // le mode Rapide (FreeRate) est actif
  bool at_floor = false;   // au plancher ×1 alors que l'écran permet plus
  bool can_store = false;  // une mémoire de l'Auto existe pour cette vidéo
};

struct AutoActions {
  bool log_free = false;                 // la ligne du mode Rapide
  bool log_change = false;               // le palier a changé : sa ligne de journal
  std::optional<std::int64_t> floor_us;  // figé au plancher depuis tant : l'avertissement
  std::optional<int> notify;             // le palier tenu à annoncer à l'écran
  std::optional<int> store;              // le palier tenu à retenir
};

class AutoWatch {
 public:
  AutoActions turn(const AutoTurn& t);

 private:
  int shown_ = 0;
  int notified_ = 0;
  int stored_ = 0;
  std::int64_t changed_us_ = 0;
  std::int64_t floor_since_us_ = 0;
  std::optional<std::int64_t> floor_logged_us_;  // le dernier avertissement ; aucun encore
  std::int64_t free_logged_us_ = 0;
};

}  // namespace ajp::app::play
