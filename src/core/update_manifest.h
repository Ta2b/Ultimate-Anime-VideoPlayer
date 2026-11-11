// Le manifeste d'une mise à jour du lecteur (docs/ux-parametres.md, 4.5) : un petit fichier TOML publié avec
// la release, signé en Ed25519 (manifest.toml.sig, en hexadécimal) ; le lecteur ne croit rien qui n'est pas
// signé.
//
//   version = "0.5.0"
//   expires = "2027-02-27"     # au-delà, le manifeste est refusé : un ancien ne se rejoue pas
//   notes_url = "https://github.com/…/releases/tag/v0.5.0"
//   revoked_keys = ["…"]      # clés publiques (64 chiffres hexadécimaux) que le lecteur ne croit plus
//
// Le lecteur ne lit que ces champs (B26-07). Le manifeste porte aussi, pour verify-download.sh, le canal, la
// date et une section par archive ([linux-x64]… avec url, sha256, size) : il les ignore, de sorte qu'une
// section abîmée pour une plateforme ne prive personne de sa notification.
#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"

namespace ajp::core {

struct UpdateManifest {
  std::string version;              // « 0.5.0 » ou « 0.5.0-rc.1 »
  std::chrono::sys_days expires{};  // jour UTC après lequel le manifeste ne vaut plus
  std::string notes_url;
  std::vector<std::string> revoked_keys;  // clés publiques retirées, en hexadécimal minuscule
};

// Refus pour un TOML mal formé, une version illisible, une clé révoquée qui n'est pas hexadécimale sur 64
// chiffres, une notes_url qui n'est pas en https, un expires absent ou illisible (AAAA-MM-JJ).
std::expected<UpdateManifest, Refusal> parse_update_manifest(std::string_view toml_text);
// Le manifeste a passé son expiration (today en UTC) : un manifeste signé ancien, rejoué pour geler les mises
// à jour, ne vaut plus.
inline bool manifest_expired(const UpdateManifest& manifest, std::chrono::sys_days today) {
  return today > manifest.expires;
}

// Les versions « majeure.mineure.correctif[-préversion.N] » : <0, 0, >0. Une préversion précède sa version
// finale (0.5.0-rc.2 < 0.5.0) ; deux préversions de même début se comparent par leur nombre final. nullopt si
// l'un des textes n'est pas une version.
std::optional<int> compare_versions(std::string_view a, std::string_view b);

}  // namespace ajp::core
