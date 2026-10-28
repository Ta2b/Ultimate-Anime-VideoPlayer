// Le manifeste d'une mise à jour du lecteur (docs/ux-parametres.md, 4.5) : un petit fichier TOML publié avec
// la release, signé en Ed25519 (manifest.toml.sig, en hexadécimal) ; le lecteur ne croit rien qui n'est pas
// signé.
//
//   version = "0.5.0"
//   channel = "stable"          # ou "preview"
//   date = "2026-10-30"
//   notes_url = "https://github.com/…/releases/tag/v0.5.0"
//   [linux-x64]
//   url = "https://…/ajp-linux-x64.tar.gz"
//   sha256 = "…"                # 64 chiffres hexadécimaux
//   size = 123456789
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>

#include "core/engines.h"
#include "core/types.h"

namespace ajp::core {

struct UpdateArtifact {
  std::string url;
  std::string sha256;
  std::uint64_t size = 0;
};

struct UpdateManifest {
  std::string version;  // « 0.5.0 » ou « 0.5.0-preview.1 »
  std::string channel;  // « stable » ou « preview »
  std::string date;
  std::string notes_url;
  std::optional<UpdateArtifact> linux_x64;
  std::optional<UpdateArtifact> windows_x64;
  std::optional<UpdateArtifact> macos_arm64;
};

// Refus pour un TOML mal formé, une version illisible, un canal inconnu, un sha256 qui n'est pas hexadécimal
// sur 64 chiffres, une adresse qui n'est pas en https.
std::expected<UpdateManifest, Refusal> parse_update_manifest(std::string_view toml_text);

// Le fichier pour ce système ; rien si la release n'en livre pas.
const UpdateArtifact* artifact_for(const UpdateManifest& manifest, Os os);

// Les versions « majeure.mineure.correctif[-préversion.N] » : <0, 0, >0. Une préversion précède sa version
// finale (0.5.0-preview.2 < 0.5.0). nullopt si l'un des textes n'est pas une version.
std::optional<int> compare_versions(std::string_view a, std::string_view b);

}  // namespace ajp::core
