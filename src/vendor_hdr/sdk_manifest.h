// Manifeste du SDK choisi (current.toml) : quelle copie, quelle version, quels fichiers.
// Lecture et écriture pures, sur du texte ; le magasin s'occupe des fichiers.
#pragma once

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

#include "vendor_hdr/sdk_kind.h"
#include "vendor_hdr/sdk_version.h"

namespace ajp::vendor_hdr {

// Format du manifeste. Un format plus récent vient d'un lecteur plus récent : refusé.
inline constexpr std::int64_t kManifestFormat = 1;

struct ManifestFile {
  std::string path;          // relatif à la copie, séparateurs « / »
  std::uintmax_t bytes = 0;  // taille à l'import ; un écart veut dire fichier modifié
  friend bool operator==(const ManifestFile&, const ManifestFile&) = default;
};

struct Manifest {
  SdkKind kind = SdkKind::RtxVideo;
  SdkVersion version;
  std::string copy;  // dossier de la copie, relatif au magasin : « versions/1.3.0.0-<n> »
  std::vector<ManifestFile> files;
  friend bool operator==(const Manifest&, const Manifest&) = default;
};

struct ManifestProblem {
  bool newer_format = false;  // écrit par un lecteur plus récent
  std::string reason;
};

std::string format_manifest(const Manifest& manifest);

// Refuse un TOML invalide, un autre SDK que expected, une copie hors de versions/, un chemin
// qui remonte (« .. ») ou absolu.
std::expected<Manifest, ManifestProblem> parse_manifest(std::string_view text, SdkKind expected);

}  // namespace ajp::vendor_hdr
