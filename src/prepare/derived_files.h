// Ce que la dérivation d'un modèle coupé (RIFE, DistilDRBA fusionnés) vérifie sur les morceaux déjà écrits :
// leur taille et leur SHA-256, rangés dans le manifeste (C04-06). Un morceau tronqué (disque, antivirus,
// copie partielle d'un dossier de données) n'est plus gardé « parce qu'il existe » : il est réécrit, au lieu
// de faire échouer la construction du moteur à chaque lancement (B04-02). Interne à prepare/.
#pragma once

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "core/utf8_path.h"
#include "prepare/sha256.h"

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

// « <hex> <hex> … » : les SHA-256 d'une ligne « digests » du manifeste ; nullopt si un mot n'en est pas un.
inline std::optional<std::vector<std::string>> parse_digests(std::string_view rest) {
  std::vector<std::string> digests;
  while (!rest.empty()) {
    const std::size_t end = std::min(rest.find(' '), rest.size());
    const std::string_view word = rest.substr(0, end);
    if (!is_sha256_hex(word)) return std::nullopt;
    digests.emplace_back(word);
    rest.remove_prefix(std::min(end + 1, rest.size()));
  }
  return digests;
}

// Les octets d'un morceau, lus comme du texte pour le SHA-256.
inline std::string_view byte_text(const std::vector<std::byte>& bytes) {
  return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

// Le morceau écrit est-il celui dont on a la taille et l'empreinte ? La taille ne suffit pas : une corruption
// à taille constante (secteur défectueux, restauration partielle d'un dossier de données) n'a pas à rester
// (C04-06).
inline bool file_matches(const std::filesystem::path& file, std::uintmax_t size, const std::string& digest) {
  if (!file_has_size(file, size)) return false;
  const auto actual = sha256_file_known(file);
  return actual && *actual == digest;
}

// Tous les morceaux d'un manifeste sont-ils là avec la taille et le SHA-256 qu'il a rangés ? Un manifeste
// sans tailles ni empreintes (d'avant) ne se croit pas : il est refait, et les morceaux justes sont gardés.
template <class Manifest>
bool pieces_intact(const std::filesystem::path& dir, const Manifest& manifest) {
  if (manifest.sizes.empty() || manifest.sizes.size() != manifest.pieces.size()) return false;
  if (manifest.digests.size() != manifest.pieces.size()) return false;
  for (std::size_t i = 0; i < manifest.pieces.size(); ++i) {
    if (!file_matches(dir / core::path_from_utf8(manifest.pieces[i]), manifest.sizes[i], manifest.digests[i]))
      return false;
  }
  return true;
}

// La ligne « digests … » d'un manifeste, ou rien quand il n'y en a pas.
inline std::string format_digests(const std::vector<std::string>& digests) {
  if (digests.empty()) return {};
  std::string text = "digests";
  for (const std::string& digest : digests) text += ' ' + digest;
  return text + '\n';
}

// La ligne « sizes … » d'un manifeste, ou rien quand il n'y en a pas.
inline std::string format_sizes(const std::vector<std::uint64_t>& sizes) {
  if (sizes.empty()) return {};
  std::string text = "sizes";
  for (const std::uint64_t size : sizes) text += ' ' + std::to_string(size);
  return text + '\n';
}

}  // namespace ajp::prepare
