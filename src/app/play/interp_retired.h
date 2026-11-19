// Les choix d'interpolation retirés du menu (docs/interpolation.md) : le menu ne propose
// plus que DistilDRBA v2 lite, RIFE 4.22 lite et RIFE 4.25. Un play.toml ou une calibration d'avant qui nomme
// un modèle ou un moteur retiré se lit encore et tombe sur le plus proche : RIFE 4.26 sur 4.25, 4.25 lite sur
// 4.22 lite, GMFSS (union seul, ou au milieu des paires fortes) sur RIFE 4.25. Les moteurs restent dans le
// code :
// --rife-model charge encore 4.26 ou 4.25 lite pour un essai.
#pragma once

#include <string>
#include <string_view>

#include "app/play/settings.h"

namespace ajp::app::play {

// Le modèle RIFE du menu le plus proche d'un modèle retiré ; les autres noms, et le nom vide (le défaut),
// tels quels.
std::string_view nearest_rife_model(std::string_view model);

// play.toml relu : modèle retiré remplacé, GMFSS ramené à RIFE 4.25.
void retire_interpolation(PlaySettings& settings);

}  // namespace ajp::app::play
