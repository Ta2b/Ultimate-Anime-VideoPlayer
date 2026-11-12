// Le texte d'une balise du conteneur (titre de piste, de chapitre) rendu sûr à afficher.
#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace ajp::media {

// Octets UTF-8 gardés au plus.
inline constexpr std::size_t kMaxTagBytes = 256;

// Les séquences UTF-8 invalides deviennent U+FFFD, les caractères de contrôle (sauts de ligne, tabulation)
// des espaces, les marques bidirectionnelles (U+200E, U+200F, U+061C, U+202A à U+202E, U+2066 à U+2069) sont
// retirées : un U+202E retournerait le libellé qui suit (C14-08). Coupé à kMaxTagBytes octets, jamais au
// milieu d'un caractère.
std::string clean_tag_text(std::string_view raw);

}  // namespace ajp::media
