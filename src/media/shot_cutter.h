// La découpe d'une vidéo en plans, à partir des mesures de chaque image : la pré-analyse du grade
// (media/shot_probe.h). Pur calcul, sans FFmpeg ni thread : les mesures entrent dans l'ordre de présentation,
// les plans finis sortent avec leurs valeurs fixes.
//
// Mesures sur l'échelle de app::play::SceneStats (app/play/chain_shots.h) : luminance
// BT.709 des canaux R, V, B encodés (0 à 1), hors bandes noires, histogramme de 64 cases.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
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

// Les seize zones d'une image en cd/m², pour le limiteur de flashs (media/flash_guard.h) : la luma moyenne de
// chaque zone passée en lumière (gamma 2,4), le tout ramené à la luminance moyenne mesurée de l'image.
std::array<float, kShotGrid * kShotGrid> zone_nits_of(const ShotFrame& frame);

// Une séquence de flashs dangereux trouvée par la pré-analyse : le plafond tient de from_us à until_us.
struct FlashSpan {
  std::int64_t from_us = 0;
  std::int64_t until_us = 0;
  friend bool operator==(const FlashSpan&, const FlashSpan&) = default;
};

// Différence entre deux images, de 0 (identiques) à 1 : histogramme et grille.
float frame_distance(const ShotFrame& a, const ShotFrame& b);

class ShotCutter {
 public:
  // cut_before du premier plan : faux pour une analyse qui part au milieu de la vidéo.
  explicit ShotCutter(bool starts_at_cut = true) : starts_at_cut_(starts_at_cut) {}

  // Une image de plus, dans l'ordre de présentation. Rend le plan qui vient de finir, s'il y en a un.
  // Un saut attend jusqu'à trois images pour être jugé : un éclair de une à trois images (tenu deux images
  // en animation sur deux) n'est pas une coupe.
  std::optional<Shot> push(const ShotFrame& frame);
  // Fin de la vidéo (ou de l'analyse) à end_us : rend le plan en cours.
  std::optional<Shot> finish(std::int64_t end_us);

 private:
  // Les sommes d'un plan (ou d'un de ses bords) : le plan en cours ne garde pas ses images, un flux sans
  // coupe de deux heures tenait 88 Mo.
  struct Totals {
    std::size_t count = 0;
    std::array<double, kShotBins> histogram{};
    double mean = 0.0, blue = 0.0, chroma = 0.0;
    std::array<double, kernels::frame_stats::kPercentiles.size()> maxrgb{};
    double apl = 0.0, maxrgb_mean = 0.0, bright = 0.0, highlights = 0.0, skin = 0.0;
    std::array<double, kShotLumBins> lum_hist{};
    float apl_min = 0.0F, apl_max = 0.0F, max_scl = 0.0F;
    void add(const ShotFrame& frame);
    ShotStats stats() const;
    ShotMeta meta() const;
  };
  struct Motion {
    std::int64_t pts_us = 0;
    float distance = 0.0F;
  };

  std::optional<Shot> judge(const ShotFrame& anchor, const ShotFrame& frame);
  void keep(const ShotFrame& frame);
  float limit(std::int64_t pts_us);
  std::optional<Shot> close_shot(std::int64_t end_us);

  bool starts_at_cut_ = true;
  bool first_shot_ = true;
  // Le plan en cours : ses sommes, son début, et ses premières et dernières images (les bords d'un fondu).
  Totals totals_;
  std::int64_t start_us_ = 0;
  std::vector<ShotFrame> head_;
  std::deque<ShotFrame> tail_;
  std::optional<ShotFrame> anchor_;  // la dernière image du plan qui n'est pas un éclair
  // Les images qui ont sauté loin de l'ancre et attendent d'être jugées : coupe ou éclair.
  std::vector<ShotFrame> jumped_;
  float jump_ = 0.0F;  // le saut de la première, et le seuil qu'il a dépassé
  float jump_limit_ = 0.0F;
  bool jump_moves_ =
      false;  // le saut ne change que la place de la lumière : peut-être un départ de mouvement
  std::deque<Motion> recent_;  // les dernières distances entre dessins distincts, pour le seuil
};

}  // namespace ajp::media
