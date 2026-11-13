// L'onglet Moteurs du menu des paramètres : le moteur de SPAN et RIFE parmi ceux installés
// (engine_switch.h), l'écran des moteurs rouvert sans relancer le lecteur, les moteurs construits.
#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "app/play/input.h"
#include "app/play/settings_panel.h"
#include "core/engines.h"

namespace ajp::app::play {

// Le chemin tel que l'Explorateur le donne (« Copier comme chemin » met des guillemets) : blancs et
// guillemets doubles de part et d'autre retirés (C23-08).
std::string clean_folder_text(std::string_view typed);

// draft : le moteur coché et pas encore appliqué ; rien : celui en service. « Utiliser ce moteur »
// pousse UseEngine dans actions, « Installer d'autres moteurs » OpenEngineScreen.
// engines_dir : le réglage « Dossier d'installation des moteurs » (play.toml, engines.install_dir) ; folder,
// son champ de saisie ; fast_precision, « Vitesse avant précision » (play.toml, engines.fast_precision).
void engines_tab(const SettingsChoices& choices, std::optional<core::EngineId>& draft,
                 std::vector<Input>& actions, std::string& engines_dir, FolderField& folder,
                 bool& fast_precision);

// En tête de Vidéo > Post-traitement : les étages que la chaîne joue sans eux (SPAN, interpolation refusés ou
// annulés, cran ramené à Équilibre), l'interpolation coupée faute de carte (motion_cut.h) et « Réessayer »,
// qui pousse RetryStages. Rien quand la chaîne demandée tourne.
void stages_note(const SettingsChoices& choices, std::vector<Input>& actions);

}  // namespace ajp::app::play
