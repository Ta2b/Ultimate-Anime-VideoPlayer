// Les pages Couleur du menu des paramètres : la sortie HDR (pic de l'écran) et le grade (saturation,
// contraste, éclats, noirs, look). Sorties de settings_panel.cpp pour que celui-ci reste lisible (C23-14).
#pragma once

#include "app/play/settings.h"
#include "app/play/settings_panel.h"
#include "core/profile.h"

namespace ajp::app::play {

// Un curseur sur un réglage du grade (double) ; true s'il a bougé. La valeur revient arrondie au centième et
// bornée.
bool grade_slider(const char* label, double& value, core::Range<double> range);
void hdr_tab(PlaySettings& d, const SettingsChoices& choices);
void grade_tab(PlaySettings& d);

}  // namespace ajp::app::play
