// A29-02, A33-03, décision 7 : un trou dans le temps recale les horloges par un seek exact à l'image d'avant
// ; une veille met aussi la lecture en pause, avec un message. Un gel de la boucle (barre de titre tenue sous
// Windows, SIGSTOP) n'est pas une veille : ni pause ni message (B31-13). L'appareil perdu pendant le trou
// arrête la lecture sur DeviceLost (relance en pause, A31-02, C31-01). Après une veille, annoncée
// (sleep_notice.h) ou vue au réveil seulement (C31-08), la chaîne se rouvre : sous Linux, la mémoire vidéo n'est pas toujours gardée (NVreg_PreserveVideoMemoryAllocations=0)
// et ses tampons perdus ne disent rien (B31-01). Un autre refus du GPU au réveil rouvre la chaîne aussi, au
// lieu d'arrêter la vidéo (B31-18).
#pragma once

#include <optional>

#include "app/play/chain_setup.h"
#include "app/play/session.h"
#include "app/play/sleep_watch.h"
#include "app/play/ui.h"
#include "core/types.h"

namespace ajp::app::play {

struct AfterGap {
  std::optional<Refusal> lost;  // l'appareil est perdu : la relance
  bool reopen = false;
  bool paused = false;  // une veille : la lecture est en pause, y compris dans la relance qui suit
};
// Le tour de boucle après un trou. announced : une veille annoncée avant le trou (remis à faux ici).
AfterGap after_gap(Session& session, const ChainEngines& c, PlayerUi& ui, SleepWatch& watch,
                   const WindowEvents& window, bool& announced);

}  // namespace ajp::app::play
