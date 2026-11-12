// Les copies « <nom>.invalide-AAAAMMJJ-HHMMSS » d'un fichier de réglages mis de côté : sans plafond, un
// disque qui abîme play.toml à chaque coupure en laisserait une par séance (C36-10).
#pragma once

#include <cstddef>
#include <filesystem>

namespace ajp::app::play {

inline constexpr std::size_t kKeptInvalidCopies = 3;

// Ne garde que les kept copies les plus récentes de path (l'horodatage du nom se trie comme la date).
void prune_invalid_copies(const std::filesystem::path& path, std::size_t kept = kKeptInvalidCopies);

}  // namespace ajp::app::play
