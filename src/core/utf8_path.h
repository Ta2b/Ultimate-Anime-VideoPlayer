// Chemins en UTF-8, quel que soit le système (P5.12) : la seule paire de conversions du projet.
//
// Sous Windows, path::string() passe par la page de code ANSI et lève une exception sur un caractère
// qu'elle ne sait pas écrire, et path(std::string) lit ses octets dans cette même page. Un chemin écrit
// dans un message, un journal, une ligne de commande, un fichier ou un appel à FFmpeg passe par
// to_utf8 ; un chemin lu de ces endroits, par path_from_utf8. tools/ci/check-path-string.sh empêche
// les path::string() de se multiplier.
#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace ajp::core {

inline std::string to_utf8(const std::filesystem::path& path) {
  const std::u8string text = path.u8string();
  return {text.begin(), text.end()};
}

inline std::filesystem::path path_from_utf8(std::string_view text) {
  return {std::u8string(text.begin(), text.end())};
}

}  // namespace ajp::core
