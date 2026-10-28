// L'onglet Moteurs du menu des paramètres : le moteur de SPAN et RIFE parmi ceux installés
// (engine_switch.h), l'écran des moteurs rouvert sans relancer le lecteur, les moteurs construits.
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "app/play/input.h"
#include "app/play/settings_panel.h"
#include "core/engines.h"

namespace ajp::app::play {

// draft : le moteur coché et pas encore appliqué ; rien : celui en service. « Utiliser ce moteur »
// pousse UseEngine dans actions, « Installer d'autres moteurs » OpenEngineScreen.
// engines_dir : le réglage « Dossier d'installation des moteurs » (play.toml, engines.install_dir).
void engines_tab(const SettingsChoices& choices, std::optional<core::EngineId>& draft,
                 std::vector<Input>& actions, std::string& engines_dir);

}  // namespace ajp::app::play
