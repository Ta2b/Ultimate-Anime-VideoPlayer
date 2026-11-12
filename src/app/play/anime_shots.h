// Méga plan A2 (--anime-shots, actif par défaut ; la sortie SDR seule, la sortie HDR garde le relèvement fixe
// : render::steady_hdr) : la méthode « HDR anime » décide par plan. Son exposition des surfaces (an_lift) et
// sa marge des éclats (la luminance moyenne de la scène, fs_apl) sont fixées par la pré-analyse, une fois par
// plan, et changent à l'image de coupe (audit F3 : un glissé après la coupe se voyait, un changement à la
// coupe est masqué par elle). La règle de l'exposition est celle de l'ancienne courbe par plan (8.5,
// render::kAnimeExposureSlope) : 0,25 diaphragme par diaphragme d'APL du plan rapportée à la médiane de la
// série, mais bornée à |E| ≤ kAnimeSurfaceMaxE : la peau et les surfaces restent ancrées (±5 % d'un plan à
// l'autre, ≈ 3 ΔE) ; la marge des éclats suit, elle, la lumière de chaque plan.
#pragma once

#include <cstdint>
#include <vector>

#include "media/shot_cutter.h"
#include "render/grade_params.h"

namespace ajp::app::play {

inline constexpr std::int64_t kAnimeGlideUs = 250'000;  // le glissé d'un même plan mieux connu
inline constexpr double kAnimeSurfaceMaxE = 0.07;       // 2^0,07 = +5 % : surfaces ancrées entre plans

// L'exposition d'un plan, en diaphragmes ; 0 sans série connue.
double anime_exposure(double shot_apl_nits, double series_apl_nits);
// La médiane des APL des plans connus (nits) ; 0 sans plan.
double series_apl_of(const std::vector<media::Shot>& shots);

// Ce que la méthode tient d'un plan : son exposition et sa lumière.
struct AnimeShotLook {
  double e = 0.0;
  render::SceneLight light;
  friend bool operator==(const AnimeShotLook&, const AnimeShotLook&) = default;
};

// D'un plan à l'autre : à une coupe, le nouveau plan aussitôt (à l'image de coupe) ; hors coupe (le même
// plan mieux connu, la jointure de deux analyses, un fondu), un glissé de kAnimeGlideUs depuis la valeur
// montrée (B42-08).
class AnimeGlide {
 public:
  void aim(const AnimeShotLook& target, std::int64_t now_us, bool cut);
  AnimeShotLook at(std::int64_t now_us) const;
  bool gliding(std::int64_t now_us) const;
  bool known() const { return known_; }

 private:
  AnimeShotLook from_;
  AnimeShotLook to_;
  std::int64_t start_us_ = 0;
  bool known_ = false;
};

}  // namespace ajp::app::play
