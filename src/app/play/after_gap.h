// A29-02, A33-03, décision 7 : un trou dans le temps recale les horloges par un seek exact à l'image d'avant
// ; une veille met aussi la lecture en pause, avec un message. Un gel de la boucle (barre de titre tenue sous
// Windows, SIGSTOP) n'est pas une veille : ni pause ni message (B31-13). L'appareil perdu pendant le trou
// arrête la lecture sur DeviceLost (relance, A31-02). Après une veille annoncée (sleep_notice.h), la chaîne
// se rouvre : sous Linux, la mémoire vidéo n'est pas toujours gardée (NVreg_PreserveVideoMemoryAllocations=0)
// et ses tampons perdus ne disent rien (B31-01). Un autre refus du GPU au réveil rouvre la chaîne aussi, au
// lieu d'arrêter la vidéo (B31-18).
#pragma once

#include <cstdint>
#include <expected>
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
};
// Le tour de boucle après un trou. announced : une veille annoncée avant le trou (remis à faux ici).
AfterGap after_gap(Session& session, const ChainEngines& c, PlayerUi& ui, SleepWatch& watch,
                   const WindowEvents& window, bool& announced);
// Le même tour, le trou déjà mesuré (testé seul, C12-09) : gap_us, sa durée ; sleep_seen : SleepWatch y a vu
// une veille ; health : l'état du GPU au réveil (rien : pas de backend) ; now_us, l'instant du message.
AfterGap after_gap_seen(Session& session, Controls& controls, std::int64_t gap_us, bool sleep_seen,
                        const std::optional<std::expected<void, Refusal>>& health, bool& announced,
                        std::int64_t now_us);

}  // namespace ajp::app::play
