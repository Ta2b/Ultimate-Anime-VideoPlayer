// Les choix d'interpolation retirés du menu (docs/interpolation.md) : le menu ne propose
// plus que DistilDRBA v2 lite, RIFE 4.22 lite et RIFE 4.25. Un play.toml ou une calibration d'avant qui nomme
// un modèle ou un moteur retiré se lit encore et tombe sur le plus proche : RIFE 4.26 sur 4.25, 4.25 lite sur
// 4.22 lite, GMFSS (union seul, ou au milieu des paires fortes) sur RIFE 4.25. Les moteurs restent dans le
// code :
// --rife-model charge encore 4.26 ou 4.25 lite pour un essai.
//
// GMFSS revient avec le natif, par défaut (docs/gmfss-natif.md, étape 5) : le lecteur le dit avant toute
// lecture de play.toml (keep_gmfss), et GMFSS union, seul ou au milieu des paires fortes, reste alors tel
// quel dans play.toml et au menu. Sous --no-gmfss-native, il tombe sur RIFE 4.25.
#pragma once

#include <string>
#include <string_view>

#include "app/play/settings.h"

namespace kirari::app::play {

// Le modèle RIFE du menu le plus proche d'un modèle retiré ; les autres noms, et le nom vide (le défaut),
// tels quels.
std::string_view nearest_rife_model(std::string_view model);

// play.toml relu : modèle retiré remplacé, GMFSS ramené à RIFE 4.25 sauf sous keep_gmfss(true).
void retire_interpolation(PlaySettings& settings);

// GMFSS gardé dans play.toml et proposé au menu (le natif, PlayOptions::gmfss_native) : posé une fois au
// lancement du lecteur, avant toute lecture des réglages ; éteint tant qu'il n'est pas posé (les tests).
void keep_gmfss(bool keep);
bool gmfss_kept();

}  // namespace kirari::app::play
