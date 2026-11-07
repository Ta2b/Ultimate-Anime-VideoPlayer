#pragma once

// Les petits outils de texte partagés (P5.13) : un seul trim, une seule mise en minuscules. ASCII seulement,
// sans dépendre de la locale (std::isspace et std::tolower la lisent) : ce qu'on compare ici, ce sont des
// clés, des balises de langue, des extensions et des lignes de journal.
// tools/ci/check-utilities.sh refuse une copie locale.

#include <cstddef>
#include <string>
#include <string_view>

namespace ajp::core {

constexpr bool is_ascii_space(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

constexpr char ascii_lower(char c) {
  return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c;
}

constexpr char ascii_upper(char c) {
  return c >= 'a' && c <= 'z' ? static_cast<char>(c - 'a' + 'A') : c;
}

// text sans les blancs du début et de la fin.
constexpr std::string_view trim(std::string_view text) {
  while (!text.empty() && is_ascii_space(text.front())) text.remove_prefix(1);
  while (!text.empty() && is_ascii_space(text.back())) text.remove_suffix(1);
  return text;
}

// a et b égaux sans égard à la casse ASCII.
constexpr bool iequal(std::string_view a, std::string_view b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (ascii_lower(a[i]) != ascii_lower(b[i])) return false;
  return true;
}

// text en minuscules ASCII (les octets UTF-8 au-delà de 127 restent tels quels).
inline std::string ascii_lower(std::string_view text) {
  std::string out(text);
  for (char& c : out) c = ascii_lower(c);
  return out;
}

}  // namespace ajp::core
