// Quand un refus pendant la lecture retire un étage au lieu d'arrêter la vidéo (B31-02, C31-05).
#pragma once

#include "app/play/chain_setup.h"
#include "core/types.h"

namespace ajp::app::play {

// B31-02 : un refus de l'étage GPU pendant la lecture (enqueue, mémoire) retire un étage ; la chaîne se
// rouvre au même endroit sans lui, avec l'avis et « Réessayer ». Un appareil perdu reste une perte.
// C31-05 : un refus de l'affichage (mémoire vidéo saturée, pl_render_image refusé) sur un appareil sain
// retire aussi un étage, l'interpolation d'abord, au lieu d'arrêter la vidéo.
bool retreats(const Refusal& failure, const ChainEngines& c);

}  // namespace ajp::app::play
