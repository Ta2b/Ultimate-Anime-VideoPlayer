// Les onglets Image (SPAN, réduction) et Fluidité (interpolation) du menu des paramètres, qui règlent la
// chaîne : la lecture les montre, et l'onglet Export (settings_export.h) les reprend pour la chaîne de
// l'export.
#pragma once

#include "app/play/settings.h"
#include "app/play/settings_panel.h"

namespace ajp::app::play {

void image_tab(PlaySettings& d, const SettingsChoices& choices);
// exporting : la chaîne d'un export, sans ce qui ne vaut qu'à l'écran (limites de l'écran et du direct,
// Smooth Motion).
void fluidity_tab(PlaySettings& d, const SettingsChoices& choices, bool exporting = false);

}  // namespace ajp::app::play
