// La source et l'écran tels que le lecteur les voit : taille, cadence, HDR, luminance, et d'où
// vient chaque valeur de l'écran.
#pragma once

#include <cstdint>

#include "core/profile.h"
#include "core/types.h"

namespace ajp::core {

struct SourceInfo {
  int width = 0;
  int height = 0;
  double fps = 0.0;
};

// D'où vient un champ de DisplayInfo, pour l'OSD et le journal : la valeur par défaut de
// ce fichier, un réglage du profil, une mesure du système, ou --display.
enum class Provenance : std::uint8_t { Default, Profile, System, CommandLine };

// Espace de couleur de la sortie présentée.
enum class OutputSpace : std::uint8_t {
  Srgb,   // SDR : sRGB, primaires BT.709
  Pq,     // HDR10 : PQ (ST 2084), primaires BT.2020
  ScRgb,  // linéaire étendu, primaires BT.709 : le compositeur HDR de Windows
};

namespace display_defaults {
// Blanc de référence BT.2408, celui de libplacebo (PL_COLOR_SDR_WHITE) ; un écran SDR
// inconnu vaut ce blanc en crête, avec un contraste de 1000:1.
inline constexpr double kSdrWhiteNits = 203.0;
inline constexpr double kPeakNits = 203.0;
inline constexpr double kMinNits = 0.203;
}  // namespace display_defaults

namespace limits {
inline constexpr Range<double> kPeakNits{1.0, 10000.0};
inline constexpr Range<double> kSdrWhiteNits{1.0, 10000.0};
}  // namespace limits

struct DisplayProvenance {
  Provenance size = Provenance::Default;
  Provenance refresh = Provenance::Default;
  Provenance hdr = Provenance::Default;
  Provenance vrr = Provenance::Default;
  Provenance peak = Provenance::Default;
  Provenance min = Provenance::Default;
  Provenance sdr_white = Provenance::Default;
  Provenance space = Provenance::Default;
  friend bool operator==(const DisplayProvenance&, const DisplayProvenance&) = default;
};

struct DisplayInfo {
  int width = 0;
  int height = 0;
  // Fixe : le rafraîchissement ; VRR : le plafond du panneau ; 0 sans VRR : écran inconnu.
  double refresh_hz = 0.0;
  bool hdr = false;  // HDR actif : la sortie part en HDR maintenant
  bool vrr = false;  // rafraîchissement variable actif
  double peak_nits = display_defaults::kPeakNits;
  double min_nits = display_defaults::kMinNits;  // dans [0, peak_nits[
  double sdr_white_nits = display_defaults::kSdrWhiteNits;
  OutputSpace space = OutputSpace::Srgb;
  DisplayProvenance provenance;
};

}  // namespace ajp::core
