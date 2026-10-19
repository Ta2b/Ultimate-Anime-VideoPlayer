// Métadonnées HDR10 de l'export : affichage de mastering (SMPTE ST 2086) et niveaux de
// lumière (CTA-861.3, MaxCLL et MaxFALL).
//
// Le pic de mastering est le pic visé par le grade. MaxCLL et MaxFALL se mesurent pendant
// l'encodage, image par image, et s'écrivent au mux final : le conteneur ne les connaît
// qu'après la dernière image. Aucune dépendance à FFmpeg dans cet en-tête.
#pragma once

#include <cstdint>
#include <expected>
#include <vector>

#include "core/types.h"
#include "export/frame_source.h"

namespace ajp::exporter {

struct Chromaticity {
  double x = 0.0;
  double y = 0.0;
};

// L'affichage de mastering : primaires BT.2020 et point blanc D65 par défaut.
struct Hdr10Mastering {
  Chromaticity red{.x = 0.708, .y = 0.292};
  Chromaticity green{.x = 0.170, .y = 0.797};
  Chromaticity blue{.x = 0.131, .y = 0.046};
  Chromaticity white{.x = 0.3127, .y = 0.3290};
  double max_nits = 1000.0;  // pic visé par le grade, cd/m²
  double min_nits = 0.0001;  // noir de l'affichage, cd/m²
};

// Bornes de validation : celles des champs SEI et des boîtes mdcv / clli.
inline constexpr double kMinMasteringPeak = 1.0;
inline constexpr double kMaxMasteringPeak = 10000.0;
inline constexpr double kMaxMasteringBlack = 50.0;

// Refus FieldOutOfRange : chromaticité hors de [0, 1], pic hors de [1, 10000], noir négatif,
// au-delà de 50 cd/m² ou au moins égal au pic.
std::expected<void, Refusal> validate_mastering(const Hdr10Mastering& mastering);

// Niveaux de lumière du flux, en cd/m² entiers.
struct LightLevel {
  std::uint32_t max_cll = 0;   // plus grande composante R, G ou B d'un pixel sur tout le flux
  std::uint32_t max_fall = 0;  // plus grande moyenne d'image de cette composante
  friend bool operator==(const LightLevel&, const LightLevel&) = default;
};

// Luminance en cd/m² d'un signal PQ normalisé dans [0, 1] (EOTF de SMPTE ST 2084).
double pq_to_nits(double signal);

// Mesure MaxCLL et MaxFALL image par image. La composante la plus forte d'un pixel donne sa
// lumière : la courbe PQ est croissante, donc la plus grande valeur codée suffit, lue dans une
// table de 65536 luminances calculée une fois par export.
class LightMeter {
 public:
  LightMeter();

  // Image hôte RGBA16 PQ : chaque pixel est lu, A ignoré.
  void add(const HostFrame& frame);
  // Voie appareil : la mesure faite par le backend sur le GPU.
  void add(const FrameLight& light);

  // Arrondi au cd/m² le plus proche ; 0 sans image.
  LightLevel level() const;

 private:
  std::vector<double> nits_;  // luminance de chaque valeur codée sur 16 bits
  double max_nits_ = 0.0;
  double max_average_nits_ = 0.0;
};

}  // namespace ajp::exporter
