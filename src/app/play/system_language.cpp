#include "app/play/system_language.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <sstream>
#include <utility>

#include <SDL3/SDL_locale.h>
#include <SDL3/SDL_stdinc.h>

#include "app/play/tracks.h"
#include "core/text.h"
#include "media/language.h"

namespace ajp::app::play {
namespace {

std::string g_system_language = "eng";  // NOLINT : posée une fois au démarrage, avant les réglages

std::string read_text(const std::filesystem::path& file) {
  std::ifstream in(file, std::ios::binary);
  if (!in) return {};
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

// La valeur de key dans la section [group] d'un fichier de configuration KDE (format INI) ; vide sans elle.
std::string ini_value(std::string_view text, std::string_view group, std::string_view key) {
  bool inside = false;
  while (!text.empty()) {
    const std::size_t end = text.find('\n');
    const std::string_view line = core::trim(text.substr(0, end));
    text = end == std::string_view::npos ? std::string_view{} : text.substr(end + 1);
    if (line.starts_with('[')) {
      inside = line == "[" + std::string(group) + "]";
      continue;
    }
    const std::size_t eq = line.find('=');
    if (!inside || eq == std::string_view::npos) continue;
    // « LANGUAGE[$e]=… » : KDE marque ainsi une valeur à développer ; la clé reste LANGUAGE.
    std::string_view name = core::trim(line.substr(0, eq));
    name = name.substr(0, name.find('['));
    if (name == key) return std::string(core::trim(line.substr(eq + 1)));
  }
  return {};
}

// « fr:en_US » (LANGUAGE de gettext) : chaque langue, dans l'ordre.
std::vector<std::string> split_list(std::string_view list) {
  std::vector<std::string> out;
  while (!list.empty()) {
    const std::size_t colon = list.find(':');
    const std::string_view one = core::trim(list.substr(0, colon));
    if (!one.empty()) out.emplace_back(one);
    list = colon == std::string_view::npos ? std::string_view{} : list.substr(colon + 1);
  }
  return out;
}

// La langue primaire d'une balise, en minuscules : « fr_FR.UTF-8@euro », « fr-CA » et « FR » donnent « fr ».
std::string primary(std::string_view tag) {
  std::string out(tag.substr(0, tag.find_first_of("-_.@")));
  out = core::ascii_lower(out);
  return out;
}

}  // namespace

std::vector<std::string> kde_locales(std::string_view plasma_localerc, std::string_view kdeglobals) {
  std::vector<std::string> out = split_list(ini_value(plasma_localerc, "Translations", "LANGUAGE"));
  if (out.empty()) out = split_list(ini_value(kdeglobals, "Locale", "Language"));
  return out;
}

std::vector<std::string> system_locales(const std::filesystem::path& home) {
  std::vector<std::string> out;
  if (!home.empty()) {
    const std::filesystem::path config = home / ".config";
    out = kde_locales(read_text(config / "plasma-localerc"), read_text(config / "kdeglobals"));
  }
  int count = 0;
  if (SDL_Locale** locales = SDL_GetPreferredLocales(&count); locales != nullptr) {
    for (int i = 0; i < count; ++i) {
      if (locales[i] == nullptr || locales[i]->language == nullptr) continue;
      std::string tag = locales[i]->language;
      if (locales[i]->country != nullptr) tag += std::string("_") + locales[i]->country;
      out.push_back(std::move(tag));
    }
    SDL_free(reinterpret_cast<void*>(locales));  // un seul bloc, pointeurs compris
  }
  return out;
}

std::string pick_menu_language(std::span<const std::string> locales) {
  for (const std::string& tag : locales) {
    const std::string code = primary(tag);
    if (code.empty() || code == "c" || code == "posix") continue;
    const std::string_view canonical = media::canonical_language(code);
    for (const Language& language : menu_languages())
      if (language.code == canonical) return std::string(language.code);
  }
  return "eng";
}

void set_system_language(std::string code) {
  g_system_language = std::move(code);
}

std::string_view system_language() {
  return g_system_language;
}

std::vector<media::LanguagePreset> default_language_presets() {
  const std::string own = g_system_language.empty() ? std::string("eng") : g_system_language;
  // Japonais : la version originale, sans sous-titres ; la seconde préférence prend l'audio du fichier.
  if (own == "jpn") return {{"jpn", ""}, {"", ""}};
  return {{"jpn", own}, {own, ""}};
}

}  // namespace ajp::app::play
