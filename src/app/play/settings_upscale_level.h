// La liste « Agrandissement » en tête de l'onglet Vidéo > Agrandissement : Désactivé, puis quatre niveaux, du
// plus léger au plus fin : Anime4K, ArtCNN, AnimeJaNai Performance et Balanced. Sharp1 et le FP8 se cochent à
// côté, pour AnimeJaNai seulement. Rien de neuf dans play.toml : le niveau se lit de `upscale` et `model`.
#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "app/play/settings.h"
#include "app/play/settings_panel.h"

namespace ajp::app::play {

enum class UpscaleLevel : std::uint8_t { VeryLow, Low, Medium, High };

// Le niveau d'un modèle ; nullopt : un modèle hors catalogue (--model, un essai).
std::optional<UpscaleLevel> upscale_level(std::string_view model);
// Le modèle d'un niveau ; sharp : la variante Sharp1 quand le niveau en a une.
std::string_view upscale_level_model(UpscaleLevel level, bool sharp);

// La liste et son infobulle ; elle allume ou éteint l'agrandissement et pose le modèle.
void upscale_combo(PlaySettings& d, const SettingsChoices& choices);
// La variante Sharp1 d'AnimeJaNai ; grisée hors Medium et High.
void sharp_checkbox(PlaySettings& d, const SettingsChoices& choices);
// Le FP8 de SPAN ; grisé hors AnimeJaNai et hors carte qui le calcule (RTX 40 et plus).
void fp8_checkbox(PlaySettings& d, const SettingsChoices& choices);
bool is_sharp(std::string_view model);
// L'agrandissement tourne : allumé, avec un modèle (rien d'installé : Désactivé).
bool upscale_on(const PlaySettings& d, const SettingsChoices& choices);
// Le FP8 tourne : l'agrandissement est allumé, sur AnimeJaNai, en précision Performance, sur une carte qui
// le calcule.
bool fp8_in_use(const PlaySettings& d, const SettingsChoices& choices);

}  // namespace ajp::app::play
