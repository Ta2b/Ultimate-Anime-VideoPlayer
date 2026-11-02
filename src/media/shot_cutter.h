// La découpe d'une vidéo en plans, à partir des mesures de chaque image : la pré-analyse du grade adaptatif
// (media/shot_probe.h). Pur calcul, sans FFmpeg ni thread : les mesures entrent dans l'ordre de présentation,
// les plans finis sortent avec leurs valeurs fixes.
//
// Mesures sur l'échelle de gpu::KernelOps::SceneStats (gpu/kernel_ops.h) : luminance
// BT.709 des canaux R, V, B encodés (0 à 1), hors bandes noires, histogramme de 64 cases.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace ajp::media {

inline constexpr int kShotBins = 64;
inline constexpr int kShotGrid = 4;  // grille kShotGrid × kShotGrid de luminances moyennes

// Les mesures d'une image.
struct ShotFrame {
  std::int64_t pts_us = 0;
  std::array<float, kShotBins> histogram{};  // parts de l'image (somme 1)
  std::array<float, kShotGrid * kShotGrid> grid{};
  float mean = 0.0F;
  float blue = 0.0F;    // moyenne de max(0, B - max(R, V))
  float chroma = 0.0F;  // moyenne de max(R, V, B) - min(R, V, B)
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
};

// Les centiles et les parts d'un histogramme de kShotBins cases.
ShotStats stats_of(const std::array<float, kShotBins>& histogram, float mean, float blue, float chroma);

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
};

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
