// La langue du système, pour les langues préférées par défaut (audio japonais et sous-titres dans la langue
// du système, puis audio dans la langue du système sans sous-titres). Lue une fois au démarrage
// (set_system_language) : sous KDE Plasma dans plasma-localerc (Paramètres du système > Langue), puis par SDL
// (Windows : les langues d'affichage de l'utilisateur ; Linux et macOS : celles de la session).
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "app/play/i18n.h"
#include "media/selection.h"

namespace kirari::app::play {

// Les langues de l'utilisateur, la préférée d'abord, en balises (« fr_FR », « en-US », « fr »). home : le
// dossier de l'utilisateur, où KDE range ~/.config/plasma-localerc ; vide, seul SDL est lu.
std::vector<std::string> system_locales(const std::filesystem::path& home);

// Les langues de KDE Plasma lues dans le texte de plasma-localerc ([Translations] LANGUAGE, « fr:en_US »)
// ou, à défaut, de kdeglobals ([Locale] Language) ; vide sans elles.
std::vector<std::string> kde_locales(std::string_view plasma_localerc, std::string_view kdeglobals);

// Le code (ISO 639-2/B, « fre ») de la première langue de locales que les menus des langues préférées
// proposent ; « eng » sans aucune.
std::string pick_menu_language(std::span<const std::string> locales);

// Le choix simple des langues (Langues > Langues, assistant) : la version originale sous-titrée dans sa
// langue, ou la version doublée dans sa langue ; Custom : les préférences réglées une à une (Avancé).
enum class LanguageVersion : std::uint8_t { Original, Dubbed, Custom };
inline constexpr std::size_t kLanguageVersionCount = 3;
std::string_view version_name(LanguageVersion version);  // « original », « dubbed », « custom » (play.toml)
std::optional<LanguageVersion> version_from_name(std::string_view name);

// Les préférences de pistes d'une langue (code ISO 639-2/B, « fre ») et d'une version. L'originale : la piste
// marquée originale par le fichier, puis le japonais, puis le chinois (les animés chinois), sous-titrés dans
// sa langue, puis le japonais sous-titré en anglais, enfin sa langue doublée sans sous-titres. La doublée :
// sa langue sans sous-titres d'abord, puis la même suite que l'originale. Un audio dans sa propre langue ne
// prend jamais de sous-titres. Custom n'a pas de suite à elle : celle de l'originale.
std::vector<media::LanguagePreset> language_presets_for(std::string_view own, LanguageVersion version);
// Les deux préférences par défaut d'avant le choix simple ({jpn, X}, {X, ""}) : la langue X, pour relire un
// ancien play.toml comme l'originale dans cette langue ; rien pour d'autres préférences.
std::optional<std::string> legacy_default_language(const std::vector<media::LanguagePreset>& presets);

// La langue du système retenue au démarrage (« eng » tant qu'aucune ne l'a été) et les préférences de
// langue par défaut qui en découlent (l'originale dans cette langue).
void set_system_language(std::string code);
std::string_view system_language();
std::vector<media::LanguagePreset> default_language_presets();
// La langue de l'interface que suit le réglage « system » (le défaut) : « fr » sur un système en français,
// « en » sinon, l'anglais pour toute langue que l'interface n'a pas (A19-03).
std::string default_interface_language();
// Le réglage de la langue de l'interface (interface.language de play.toml) qui suit celle du système.
inline constexpr std::string_view kSystemInterfaceLanguage = "system";
// La langue montrée pour un réglage « system », « en » ou « fr » : « system » prend celle du système.
i18n::Language interface_language(std::string_view setting);
// « system », « en » ou « fr » : ce que play.toml peut dire de la langue de l'interface.
bool known_interface_language(std::string_view setting);

}  // namespace kirari::app::play
