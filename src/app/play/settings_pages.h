// Le contenu des onglets Général > Application, Général > Statistiques, Vidéo > Général et Langues du menu
// des paramètres (settings_panel.cpp les place). Les onglets de la chaîne sont dans settings_chain_tabs.h,
// ceux du HDR dans settings_color_tabs.h, l'audio dans settings_sound.h, l'export dans settings_export.h.
#pragma once

#include <optional>
#include <vector>

#include "app/play/input.h"
#include "app/play/settings.h"
#include "app/play/settings_panel.h"
#include "app/play/settings_system.h"
#include "core/engines.h"

namespace kirari::app::play {

// Général > Application : la langue de l'interface, l'épisode suivant, les mises à jour, le lecteur par
// défaut, la machine, l'assistant et « Signaler un problème ».
void application_page(PlaySettings& d, const SettingsChoices& choices, UpdateNote& updates,
                      SystemNote& system, std::vector<Input>& actions);
// Général > Application > Avancé : « Tout réinitialiser ». Vrai quand il est confirmé : le panneau rouvre
// alors la vidéo avec les réglages par défaut, sans attendre « Appliquer ».
bool reset_section(PlaySettings& draft);
void statistics_page(PlaySettings& d);

// Vidéo > Général : le préréglage, le moteur, la sortie HDR, les couleurs et « Afficher les FPS ».
void video_general_page(PlaySettings& d, const SettingsChoices& choices, std::vector<Input>& actions,
                        std::optional<core::EngineId>& engine_draft);
// Vidéo > Général > Avancé : la composition finale (filtre, débandage) et l'OLED.
void display_section(PlaySettings& d, const SettingsChoices& choices);

}  // namespace kirari::app::play
