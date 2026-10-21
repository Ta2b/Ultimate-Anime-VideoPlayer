// Les deux SDK TrueHDR, et celui que chaque système emploie.
#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace ajp::vendor_hdr {

enum class SdkKind : std::uint8_t {
  RtxVideo,   // NVIDIA RTX Video SDK 1.1, Windows, chemin CUDA
  MaxineVfx,  // NVIDIA Maxine VFX 1.3 et sa fonction TrueHDR, Linux
};

// Système hôte, tel que app le décrit (core::Os n'est pas visible depuis core_types).
enum class HostOs : std::uint8_t { Windows, Linux, MacOs };

// Nom affiché : « RTX Video SDK 1.1 », « Maxine VFX 1.3 (TrueHDR) ».
std::string_view display_name(SdkKind kind);

// Le SDK TrueHDR de ce système ; aucun sous macOS.
constexpr std::optional<SdkKind> sdk_for(HostOs os) {
  switch (os) {
    case HostOs::Windows:
      return SdkKind::RtxVideo;
    case HostOs::Linux:
      return SdkKind::MaxineVfx;
    case HostOs::MacOs:
      return std::nullopt;
  }
  return std::nullopt;
}

}  // namespace ajp::vendor_hdr
