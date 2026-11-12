// Chemins en UTF-8, quel que soit le système (P5.12) : la seule paire de conversions du projet.
//
// Sous Windows, path::string() passe par la page de code ANSI et lève une exception sur un caractère
// qu'elle ne sait pas écrire, et path(std::string) lit ses octets dans cette même page. Un chemin écrit
// dans un message, un journal, une ligne de commande, un fichier ou un appel à FFmpeg passe par
// to_utf8 ; un chemin lu de ces endroits, par path_from_utf8. tools/ci/check-path-string.sh empêche
// les path::string() de se multiplier.
#pragma once

#include <exception>
#include <filesystem>
#include <string>
#include <string_view>

namespace ajp::core {

// Le nom NTFS qui porte une moitié de paire de substitution n'a pas d'UTF-8 : u8string() lève. Un message ne
// doit jamais lever, le caractère devient U+FFFD (C20-09).
inline std::string to_utf8(const std::filesystem::path& path) {
#ifdef _WIN32
  try {
    const std::u8string text = path.u8string();
    return {text.begin(), text.end()};
  } catch (const std::exception&) {
    const std::wstring& wide = path.native();
    std::string out;
    out.reserve(wide.size());
    const auto put = [&](char32_t c) {
      if (c < 0x80) {
        out += static_cast<char>(c);
      } else if (c < 0x800) {
        out += static_cast<char>(0xC0 | (c >> 6));
        out += static_cast<char>(0x80 | (c & 0x3F));
      } else if (c < 0x10000) {
        out += static_cast<char>(0xE0 | (c >> 12));
        out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (c & 0x3F));
      } else {
        out += static_cast<char>(0xF0 | (c >> 18));
        out += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (c & 0x3F));
      }
    };
    for (std::size_t i = 0; i < wide.size(); ++i) {
      const auto c = static_cast<char32_t>(wide[i]);
      const bool high = c >= 0xD800 && c <= 0xDBFF;
      const bool low = c >= 0xDC00 && c <= 0xDFFF;
      if (high && i + 1 < wide.size() && wide[i + 1] >= 0xDC00 && wide[i + 1] <= 0xDFFF) {
        put(0x10000 + ((c - 0xD800) << 10) + (static_cast<char32_t>(wide[i + 1]) - 0xDC00));
        ++i;
      } else {
        put(high || low ? char32_t{0xFFFD} : c);
      }
    }
    return out;
  }
#else
  const std::u8string text = path.u8string();
  return {text.begin(), text.end()};
#endif
}

inline std::filesystem::path path_from_utf8(std::string_view text) {
  return {std::u8string(text.begin(), text.end())};
}

}  // namespace ajp::core
