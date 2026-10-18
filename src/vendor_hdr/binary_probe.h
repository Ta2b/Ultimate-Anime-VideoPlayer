// En-têtes des bibliothèques d'un SDK : format et version, lus comme des données.
// Rien n'est chargé ni exécuté ici ; le magasin s'en sert avant toute copie ou chargement.
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>

#include "vendor_hdr/refusal.h"
#include "vendor_hdr/sdk_version.h"

namespace ajp::vendor_hdr {

enum class BinaryFormat : std::uint8_t {
  ElfX86_64,  // bibliothèque partagée Linux x64 (ET_DYN, EM_X86_64)
  PeX64,      // DLL Windows x64 (PE32+, IMAGE_FILE_MACHINE_AMD64)
  Text,       // en-tête, licence : présent et non vide suffit
};

// Refuse root/relative s'il est absent, vide ou d'un autre format ; le détail nomme relative.
// Un lien symbolique est suivi : c'est la cible qui doit convenir.
Done check_binary(const std::filesystem::path& root, const std::filesystem::path& relative,
                  BinaryFormat format);

// FileVersion du VS_FIXEDFILEINFO de la section .rsrc d'une DLL PE32+ ; rien si absent.
std::optional<SdkVersion> pe_file_version(const std::filesystem::path& file);

}  // namespace ajp::vendor_hdr
