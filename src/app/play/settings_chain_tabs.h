// Les onglets Vidéo > Agrandissement (SPAN, réduction, netteté) et Vidéo > Interpolation du menu des
// paramètres, qui règlent la chaîne : la lecture les montre, et l'onglet Export (settings_export.h) les
// reprend pour la chaîne de l'export. Chaque onglet a sa part courante et sa part avancée, dessinée sous «
// Avancé ».
#pragma once

#include "app/play/precision.h"
#include "app/play/settings.h"
#include "app/play/settings_panel.h"

namespace ajp::app::play {

void image_tab(PlaySettings& d, const SettingsChoices& choices);
// Le filtre de réduction et la précision de SPAN.
void upscale_advanced(PlaySettings& d, const SettingsChoices& choices);
// exporting : la chaîne d'un export, sans ce qui ne vaut qu'à l'écran (limites de l'écran et du direct,
// Smooth Motion).
void fluidity_tab(PlaySettings& d, const SettingsChoices& choices, bool exporting = false);
// Les moteurs fusionnés, la précision, l'ordre du warp, GMFSS et les essais.
void motion_advanced(PlaySettings& d, const SettingsChoices& choices, bool exporting = false);
// Le cran de précision d'un réseau, parmi ceux que ce poste rend ; rien quand Équilibre est le seul. Vrai si
// le choix a changé.
bool precision_combo(const char* title, PrecisionTier& tier, const TierSupport& support);

}  // namespace ajp::app::play
