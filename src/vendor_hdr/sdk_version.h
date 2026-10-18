// Version d'un SDK importé, lue dans ses fichiers sans les charger ni les exécuter.
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "core/types.h"
#include "vendor_hdr/sdk_kind.h"

namespace ajp::vendor_hdr {

struct SdkVersion {
  std::array<std::uint32_t, 4> parts{};  // majeur, mineur, correctif, build
  std::uint32_t major() const { return parts[0]; }
  std::uint32_t minor() const { return parts[1]; }
  std::string text() const;  // « 1.3.0.0 »
  friend bool operator==(const SdkVersion&, const SdkVersion&) = default;
};

// « 1.3 », « 1.3.0.21 » : deux à quatre nombres séparés par des points.
std::optional<SdkVersion> parse_version(std::string_view text);

// Macros d'un version.h à la manière des SDK Maxine (NVAUDIOEFFECTS_MAJOR_VERSION…) :
// le premier préfixe qui définit <P>_MAJOR_VERSION et <P>_MINOR_VERSION.
std::optional<SdkVersion> parse_version_header(std::string_view text);

// Nom réel versionné d'une bibliothèque ELF : « libVideoFX.so.1.3.0.0 » pour « libVideoFX.so ».
std::optional<SdkVersion> parse_versioned_name(std::string_view file_name, std::string_view library);

// Version du SDK installé sous root :
//   RTX Video : FileVersion (VS_FIXEDFILEINFO) de la DLL du SDK ;
//   Maxine : version.h à la racine ou dans include/, sinon le nom réel de lib/libVideoFX.so.
// Ces marqueurs restent à confirmer sur les archives originales (AH-324) ; un dossier sans
// marqueur est refusé, jamais chargé en version inconnue.
std::expected<SdkVersion, Refusal> read_sdk_version(SdkKind kind, const std::filesystem::path& root);

}  // namespace ajp::vendor_hdr
