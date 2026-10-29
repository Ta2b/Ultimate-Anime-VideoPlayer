// GMFSS fusionné (prepare/gmfss_split.h) : faut-il l'essayer pour cette lecture ? Il l'est quand le réglage
// le veut, que le moteur le fait et que GMFSS est le moteur du mouvement au cran Équilibre (le graphe tel que
// livré). Ce qui reste, la reconnaissance de la paire et la construction des deux moteurs, se juge en les
// tentant (player_engines.cpp) : un refus garde la paire entière.
#pragma once

#include <string>

#include "app/play/settings.h"

namespace ajp::app::play {

// Vide : GMFSS fusionné s'essaie. Sinon la raison, pour le journal.
std::string gmfss_fused_declined(const PlaySettings& settings, bool backend_does_it);

}  // namespace ajp::app::play
