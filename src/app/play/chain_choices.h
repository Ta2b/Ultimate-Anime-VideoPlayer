// Ce que le menu des paramètres sait d'une ouverture de la chaîne (player_run.cpp) : modèles, crans, moteurs,
// écran, étages refusés.
#pragma once

#include <string>

#include "app/play/chain_setup.h"
#include "app/play/player_engines.h"
#include "app/play/settings_panel.h"

namespace ajp::app::play {

SettingsChoices chain_choices(Player& p, const ChainEngines& c, const SpanPrepare& span_prepare,
                              int auto_peak);

}  // namespace ajp::app::play
