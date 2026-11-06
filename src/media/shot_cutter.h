// La découpe d'une vidéo en plans, à partir des mesures de chaque image : la pré-analyse du grade
// (media/shot_probe.h). Pur calcul, sans FFmpeg ni thread : les mesures entrent dans l'ordre de présentation,
// les plans finis sortent avec leurs valeurs fixes.
//
// Mesures sur l'échelle de gpu::KernelOps::SceneStats (gpu/kernel_ops.h) : luminance
// BT.709 des canaux R, V, B encodés (0 à 1), hors bandes noires, histogramme de 64 cases.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "kernels/spec/frame_stats.h"

namespace ajp::media {

inline constexpr int kShotBins = 64;
inline constexpr int kShotGrid = 4;              // grille kShotGrid × kShotGrid de luminances moyennes
inline constexpr std::size_t kShotLumBins = 32;  // histogramme de I (ICtCp) par code PQ, pour la courbe (8.6)

// La lumière d'une image en cd/m² (kernels/spec/frame_stats.h, échelle linéaire).
struct ShotLight {
  std::array<float, kernels::frame_stats::kPercentiles.size()> maxrgb_nits{};  // centiles de maxRGB
  float apl_nits = 0.0F;
  float maxrgb_mean_nits = 0.0F;
  float bright = 0.0F;                         // part de surface claire
  float highlights = 0.0F;                     // part de hautes lumières isolées
  float skin = 0.0F;                           // part de peau
  std::array<float, kShotLumBins> lum_hist{};  // parts de l'image par code PQ de I (somme 1)
  friend bool operator==(const ShotLight&, const ShotLight&) = default;
};

// Les mesures d'une image.
struct ShotFrame {
  std::int64_t pts_us = 0;
  std::array<float, kShotBins> histogram{};  // parts de l'image (somme 1)
  std::array<float, kShotGrid * kShotGrid> grid{};
  float mean = 0.0F;
  float blue = 0.0F;    // moyenne de max(0, B - max(R, V))
  float chroma = 0.0F;  // moyenne de max(R, V, B) - min(R, V, B)
  ShotLight light;
};

// Les valeurs d'un plan, sur l'échelle de SceneStats.
struct ShotStats {
  float mean = 0.0F;
  float p05 = 0.0F;
  float p50 = 0.0F;
  float p95 = 0.0F;
  float p995 = 0.0F;
  float dark = 0.0F;    // part sous 10 %
  float bright = 0.0F;  // part au-dessus de 80 %
  float blue = 0.0F;
  float chroma = 0.0F;
  float lit = 0.0F;       // part au-dessus de 70 % du blanc en lumière (gamma 2,4) : les grands aplats clairs
  float apl_nits = 0.0F;  // luminance moyenne, en cd/m² au blanc SDR de 203 (kernels/spec/frame_stats.h)
};

// Les centiles et les parts d'un histogramme de kShotBins cases (apl_nits laissé à 0).
ShotStats stats_of(const std::array<float, kShotBins>& histogram, float mean, float blue, float chroma);

// La lumière d'un plan (ShotMeta de l'audit) : APL minimale, moyenne et maximale, maxSCL (le plus grand
// centile 99,9 de maxRGB de ses images), et les moyennes des mesures de ses images.
struct ShotMeta {
  float apl_min_nits = 0.0F;
  float apl_mean_nits = 0.0F;
  float apl_max_nits = 0.0F;
  float max_scl_nits = 0.0F;
  ShotLight mean;
  friend bool operator==(const ShotMeta&, const ShotMeta&) = default;
};

// Un plan : de start_us (première image) à end_us (première image du plan suivant, ou fin de la vidéo).
// Un plan qui change vraiment de lumière (un fondu) garde ses valeurs du début et de la fin : le grade les
// suit image par image ; les autres, une seule valeur (first == last).
struct Shot {
  std::int64_t start_us = 0;
  std::int64_t end_us = 0;
  ShotStats first;
  ShotStats last;
  bool fade = false;
  // Vrai si le plan commence à une vraie coupe ; faux pour le premier plan analysé après un départ en
  // cours de vidéo (le début du fichier, ou un saut).
  bool cut_before = true;
  ShotMeta meta;
};

// Les mesures d'une image à pts_us, tirées de l'analyse (kernels/spec/frame_stats.h).
ShotFrame shot_frame_of(std::int64_t pts_us, const kernels::FrameStats& stats);

// Différence entre deux images, de 0 (identiques) à 1 : histogramme et grille.
float frame_distance(const ShotFrame& a, const ShotFrame& b);

class ShotCutter {
 public:
  // cut_before du premier plan : faux pour une analyse qui part au milieu de la vidéo.
  explicit ShotCutter(bool starts_at_cut = true) : starts_at_cut_(starts_at_cut) {}

  // Une image de plus, dans l'ordre de présentation. Rend le plan qui vient de finir, s'il y en a un.
  // Une image décide avec une image de retard : un éclair d'une image n'est pas une coupe.
  std::optional<Shot> push(const ShotFrame& frame);
  // Fin de la vidéo (ou de l'analyse) à end_us : rend le plan en cours.
  std::optional<Shot> finish(std::int64_t end_us);

 private:
  std::optional<Shot> close_shot(std::int64_t end_us);

  bool starts_at_cut_ = true;
  bool first_shot_ = true;
  std::vector<ShotFrame> frames_;     // le plan en cours
  std::optional<ShotFrame> anchor_;   // la dernière image du plan qui n'est pas un éclair
  std::optional<ShotFrame> pending_;  // l'image qui attend la suivante pour être jugée
  std::vector<float> recent_;         // les dernières distances, pour un seuil qui suit le mouvement
};

}  // namespace ajp::media
