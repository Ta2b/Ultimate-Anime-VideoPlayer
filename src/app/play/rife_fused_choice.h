// RIFE fusionné (kernels/spec/rife_fused.h) : faut-il l'essayer pour cette lecture ? Il l'est quand le
// réglage le veut, que le moteur le fait, que RIFE est le moteur du mouvement au cran Équilibre (le modèle
// tel que livré) et que son entrée est celle de v1 (11 plans). Ce qui reste, la reconnaissance du modèle et
// la construction des six moteurs, se juge en les tentant (chain_setup.cpp) : un refus garde le RIFE entier.
#pragma once

#include <string>

#include "app/play/settings.h"

namespace ajp::app::play {

// Vide : RIFE fusionné s'essaie. Sinon la raison, pour le journal.
std::string rife_fused_declined(const PlaySettings& settings, bool backend_does_it, int rife_planes);

}  // namespace ajp::app::play
