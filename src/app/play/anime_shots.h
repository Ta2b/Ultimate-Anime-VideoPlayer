// Méga plan A2 (essai --anime-shots) : la méthode « HDR anime » (--hdr-anime) décide par plan. Son
// exposition des surfaces (an_lift) et sa marge des éclats (la luminance moyenne de la scène, fs_apl) sont
// fixées par la pré-analyse, une fois par plan, et glissent aux coupes au lieu de sauter. La règle de
// l'exposition est celle de la courbe par plan (render/curve.h, 8.5) : 0,25 diaphragme par diaphragme d'APL
// du plan rapportée à la médiane de la série, |E| ≤ 0,25.
#pragma once

#include <cstdint>
#include <vector>

#include "media/shot_cutter.h"
#include "render/grade_params.h"

namespace ajp::app::play {

inline constexpr std::int64_t kAnimeGlideUs = 250'000;  // le glissé d'une coupe

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

// Le glissé d'un plan à l'autre : à une coupe, de ce qui était montré vers le nouveau plan en kAnimeGlideUs
// (smoothstep) ; hors coupe (le même plan, mieux connu), aussitôt.
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
