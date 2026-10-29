// Typographie française des textes de l'interface.
#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace ajp::app::play {

// L'espace insécable de la typographie française : « » et : ; ? ! % ne passent jamais seuls en début de ligne
// (ImGui ne coupe les lignes qu'aux espaces ordinaires).
inline std::string french_spaces(std::string_view text) {
  constexpr std::string_view kNbsp = "\xC2\xA0";
  std::string out;
  out.reserve(text.size() + 16);
  for (std::size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    const bool before = c == ' ' && i + 1 < text.size() &&
                        (text[i + 1] == ':' || text[i + 1] == ';' || text[i + 1] == '?' ||
                         text[i + 1] == '!' || text[i + 1] == '%');
    const bool after_open = c == ' ' && i >= 2 && text.substr(i - 2, 2) == "\xC2\xAB";  // « puis espace
    const bool before_close = c == ' ' && text.substr(i + 1, 2) == "\xC2\xBB";          // espace puis »
    if (before || after_open || before_close) {
      out += kNbsp;
    } else {
      out += c;
    }
  }
  return out;
}

}  // namespace ajp::app::play
