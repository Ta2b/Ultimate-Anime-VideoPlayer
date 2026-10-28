// DistilDRBA fusionné (kernels/spec/drba_fused.h) : faut-il l'essayer pour cette lecture ? Il l'est quand le
// réglage le veut, que le moteur le fait et que DistilDRBA est le moteur du mouvement au cran Équilibre (le
// modèle tel que livré). Ce qui reste, la reconnaissance du modèle et la construction des quatre moteurs, se
// juge en les tentant (player_engines.cpp) : un refus garde le DistilDRBA entier.
#pragma once

#include <string>

#include "app/play/settings.h"

namespace ajp::app::play {

// Vide : DistilDRBA fusionné s'essaie. Sinon la raison, pour le journal.
std::string drba_fused_declined(const PlaySettings& settings, bool backend_does_it);

}  // namespace ajp::app::play
