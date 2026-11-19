// L'onglet Langues et le choix des langues de l'assistant : le choix simple (sa langue, la version originale
// ou doublée) et, dans Avancé, les préférences de pistes une à une.
#pragma once

#include <string>
#include <vector>

#include "app/play/settings.h"

namespace kirari::app::play {

// Sa langue et la version voulue ; refait les préférences (language_presets_for) quand l'une change. En
// Custom, choisir une version revient aux préférences de cette version.
void simple_languages(std::string& own, LanguageVersion& version,
                      std::vector<media::LanguagePreset>& presets);
// Les préférences une à une, du premier choix au dernier : en changer une passe la version à Custom.
void track_choices(std::vector<media::LanguagePreset>& presets, LanguageVersion& version);

// Langues : le choix simple, la taille et le délai des sous-titres, puis Avancé. Vrai tant que le curseur du
// délai est tenu : le délai s'applique au relâchement (A23-12).
bool languages_page(PlaySettings& d);

}  // namespace kirari::app::play
