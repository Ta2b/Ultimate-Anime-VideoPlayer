// Les couleurs et le HDR du menu des paramètres : la sortie et les couleurs de Vidéo > Général (le grade
// anime, en HDR comme en SDR), et l'onglet Vidéo > HDR (pic de l'écran, blanc des surfaces, flashs ; en
// avancé, le calcul du grade). Sortis de settings_panel.cpp pour que celui-ci reste lisible (C23-14).
#pragma once

#include "app/play/settings.h"
#include "app/play/settings_panel.h"
#include "core/profile.h"

namespace ajp::app::play {

// Un curseur sur un réglage du grade (double) ; true s'il a bougé. La valeur revient arrondie au centième et
// bornée.
bool grade_slider(const char* label, double& value, core::Range<double> range);
// Vidéo > Général : la sortie HDR voulue (Auto, Activé, Désactivé) et ce qui tourne.
void output_section(PlaySettings& d, const SettingsChoices& choices);
// Vidéo > Général : saturation, contraste, éclats, noirs profonds, intensité des LUT.
void colors_section(PlaySettings& d);
void hdr_page(PlaySettings& d, const SettingsChoices& choices);

}  // namespace ajp::app::play
