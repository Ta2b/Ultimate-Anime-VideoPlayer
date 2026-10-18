// Chemin en texte UTF-8, pour les messages et les comparaisons de noms.
//
// path::string() convertit sous Windows vers la page de code ANSI et lève une exception sur un
// caractère qu'elle ne sait pas représenter (un dossier « アニメ » sous une page occidentale) :
// une exception traverserait le module. Sous Linux, les octets du nom passent tels quels.
#pragma once

#include <filesystem>
#include <string>

namespace ajp::vendor_hdr {

inline std::string utf8(const std::filesystem::path& path) {
  const std::u8string text = path.generic_u8string();
  return {text.begin(), text.end()};
}

}  // namespace ajp::vendor_hdr
