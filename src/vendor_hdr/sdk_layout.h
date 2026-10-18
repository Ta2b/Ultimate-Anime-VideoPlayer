// Contenu attendu de chaque SDK TrueHDR : fichiers exigés, parties copiées dans le dossier de
// données, version prise en charge. Une table par SDK : une nouvelle disposition d'archive se
// suit ici, et nulle part ailleurs.
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string_view>

#include "vendor_hdr/binary_probe.h"
#include "vendor_hdr/sdk_kind.h"

namespace ajp::vendor_hdr {

struct RequiredFile {
  std::string_view path;  // relatif à la racine du SDK, séparateurs « / »
  BinaryFormat format = BinaryFormat::ElfX86_64;
  bool prefix = false;       // le nom du fichier commence seulement par celui de path (« .so.1.3 »)
  std::string_view missing;  // ce que l'utilisateur doit faire s'il manque
};

struct SdkLayout {
  SdkKind kind = SdkKind::RtxVideo;
  std::string_view display_name;
  std::string_view directory;     // sous-dossier du magasin : « rtx-video », « maxine-vfx »
  std::string_view main_library;  // bibliothèque d'entrée du SDK, chargée par son chargeur
  std::span<const RequiredFile> required;
  std::span<const std::string_view> copied;  // fichiers ou dossiers copiés quand ils existent
  std::uint32_t major = 0;                   // version prise en charge : majeur et mineur
  std::uint32_t minor = 0;
};

const SdkLayout& layout(SdkKind kind);

// Chemin relatif réel d'un fichier exigé sous root : le chemin exact, ou pour un préfixe le
// premier nom du dossier, par ordre alphabétique, qui commence par lui.
std::optional<std::filesystem::path> locate(const std::filesystem::path& root, const RequiredFile& file);

}  // namespace ajp::vendor_hdr
