// Ce que la dérivation d'un modèle coupé (RIFE, DistilDRBA fusionnés) vérifie sur les morceaux déjà écrits :
// leur taille, rangée dans le manifeste. Un morceau tronqué (disque, antivirus, copie partielle d'un dossier
// de données) n'est plus gardé « parce qu'il existe » : il est réécrit, au lieu de faire échouer la
// construction du moteur à chaque lancement (B04-02). Interne à prepare/.
#pragma once

#include <charconv>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "core/utf8_path.h"

namespace ajp::prepare {

// Le fichier existe-t-il avec exactement cette taille ?
inline bool file_has_size(const std::filesystem::path& file, std::uintmax_t size) {
  std::error_code error;
  const auto actual = std::filesystem::file_size(file, error);
  return !error && size > 0 && actual == size;
}

// « 123 456 789 » : les tailles d'une ligne « sizes » du manifeste ; nullopt si un mot n'est pas un entier.
inline std::optional<std::vector<std::uint64_t>> parse_sizes(std::string_view rest) {
  std::vector<std::uint64_t> sizes;
  while (!rest.empty()) {
    std::uint64_t value = 0;
    const char* first = rest.data();
    const auto [stop, error] = std::from_chars(first, first + rest.size(), value);
    if (error != std::errc{}) return std::nullopt;
    sizes.push_back(value);
    rest.remove_prefix(static_cast<std::size_t>(stop - first));
    while (!rest.empty() && rest.front() == ' ') rest.remove_prefix(1);
  }
  return sizes;
}

// Tous les morceaux d'un manifeste sont-ils là avec la taille qu'il a rangée ? Un manifeste sans tailles
// (d'avant) ne se croit pas : il est refait.
template <class Manifest>
bool pieces_intact(const std::filesystem::path& dir, const Manifest& manifest) {
  if (manifest.sizes.empty() || manifest.sizes.size() != manifest.pieces.size()) return false;
  for (std::size_t i = 0; i < manifest.pieces.size(); ++i) {
    if (!file_has_size(dir / core::path_from_utf8(manifest.pieces[i]), manifest.sizes[i])) return false;
  }
  return true;
}

// La ligne « sizes … » d'un manifeste, ou rien quand il n'y en a pas.
inline std::string format_sizes(const std::vector<std::uint64_t>& sizes) {
  if (sizes.empty()) return {};
  std::string text = "sizes";
  for (const std::uint64_t size : sizes) text += ' ' + std::to_string(size);
  return text + '\n';
}

}  // namespace ajp::prepare
