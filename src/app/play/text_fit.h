// Copie d'un texte dans un champ de taille fixe de l'interface (nom de profil, chemin).
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <string_view>

namespace ajp::app::play {

// Le plus long préfixe de text qui tient dans max_bytes octets sans couper un caractère UTF-8.
inline std::size_t utf8_prefix(std::string_view text, std::size_t max_bytes) {
  if (text.size() <= max_bytes) return text.size();
  std::size_t n = max_bytes;
  while (n > 0 && (static_cast<unsigned char>(text[n]) & 0xC0U) == 0x80U) --n;  // octet de continuation
  return n;
}

// text dans to, terminé par '\0', tronqué à un caractère entier.
template <std::size_t N>
void copy_text(std::array<char, N>& to, std::string_view text) {
  const std::size_t n = utf8_prefix(text, N - 1);
  std::copy_n(text.data(), n, to.data());
  to[n] = '\0';
}

}  // namespace ajp::app::play
