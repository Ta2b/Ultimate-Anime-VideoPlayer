// Copie d'un SDK vers le magasin, entrée par entrée.
#pragma once

#include <filesystem>
#include <span>
#include <stop_token>
#include <string_view>

#include "vendor_hdr/refusal.h"

namespace ajp::vendor_hdr {

// Copie chaque entrée (fichier ou dossier, relative à from) qui existe vers to. Un lien
// symbolique est recréé en lien relatif s'il vise l'intérieur du SDK ; un lien qui en sort ou
// un fichier spécial est refusé. S'arrête (Cancelled) dès que stop est demandé, en laissant à
// l'appelant le soin de supprimer to.
Done copy_sdk_entries(const std::filesystem::path& from, const std::filesystem::path& to,
                      std::span<const std::string_view> entries, const std::stop_token& stop);

}  // namespace ajp::vendor_hdr
