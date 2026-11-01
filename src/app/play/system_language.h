// La langue du système, pour les langues préférées par défaut (audio japonais et sous-titres dans la langue
// du système, puis audio dans la langue du système sans sous-titres). Lue une fois au démarrage
// (set_system_language) : sous KDE Plasma dans plasma-localerc (Paramètres du système > Langue), puis par SDL
// (Windows : les langues d'affichage de l'utilisateur ; Linux et macOS : celles de la session).
#pragma once

#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "media/selection.h"

namespace ajp::app::play {

// Les langues de l'utilisateur, la préférée d'abord, en balises (« fr_FR », « en-US », « fr »). home : le
// dossier de l'utilisateur, où KDE range ~/.config/plasma-localerc ; vide, seul SDL est lu.
std::vector<std::string> system_locales(const std::filesystem::path& home);

// Les langues de KDE Plasma lues dans le texte de plasma-localerc ([Translations] LANGUAGE, « fr:en_US »)
// ou, à défaut, de kdeglobals ([Locale] Language) ; vide sans elles.
std::vector<std::string> kde_locales(std::string_view plasma_localerc, std::string_view kdeglobals);

// Le code (ISO 639-2/B, « fre ») de la première langue de locales que les menus des langues préférées
// proposent ; « eng » sans aucune.
std::string pick_menu_language(std::span<const std::string> locales);

// La langue du système retenue au démarrage (« eng » tant qu'aucune ne l'a été) et les préférences de
// langue par défaut qui en découlent.
void set_system_language(std::string code);
std::string_view system_language();
std::vector<media::LanguagePreset> default_language_presets();

}  // namespace ajp::app::play
