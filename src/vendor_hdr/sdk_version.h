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

// Macros <prefix>_MAJOR, _MINOR, _RELEASE et _BUILD d'un en-tête de version, comme
// share/videofx_version.h de Maxine VFX 1.3 (NVIDIA_VIDEOEFFECTS_SDK_VERSION_MAJOR…). Majeur et
// mineur sont exigés, les deux autres valent 0 s'ils manquent.
std::optional<SdkVersion> parse_version_header(std::string_view text, std::string_view prefix);

// Nom réel versionné d'une bibliothèque ELF : « libVideoFX.so.1.3.0.0 » pour « libVideoFX.so ».
std::optional<SdkVersion> parse_versioned_name(std::string_view file_name, std::string_view library);

// Version du SDK installé sous root :
//   RTX Video : FileVersion (VS_FIXEDFILEINFO) de la DLL du SDK (non vérifié sur l'archive) ;
//   Maxine : share/videofx_version.h, sinon le nom réel de lib/libVideoFX.so (libVideoFX.so.1.3.0) ;
//            quand les deux existent, ils doivent s'accorder, et la fonction TrueHDR
//            (NVVFXTRUEHDR_VERSION de son en-tête) doit avoir le même majeur et mineur.
// Un dossier sans marqueur est refusé, jamais chargé en version inconnue.
std::expected<SdkVersion, Refusal> read_sdk_version(SdkKind kind, const std::filesystem::path& root);

}  // namespace ajp::vendor_hdr
