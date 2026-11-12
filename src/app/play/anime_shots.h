// Méga plan A2 (--anime-shots, actif par défaut ; la sortie SDR seule, la sortie HDR garde la marge entière :
// render::steady_hdr) : la méthode « HDR anime » décide par plan. Sa marge des éclats (la luminance moyenne
// de la scène, fs_apl) est fixée par la pré-analyse, une fois par plan, et change à l'image de coupe (audit
// F3 : un glissé après la coupe se voyait, un changement à la coupe est masqué par elle). Les surfaces
// gardent le même relèvement d'un plan à l'autre : l'exposition par plan est retirée (C41-01,
// render::kAnimeLift).
#pragma once

#include <cstdint>
#include <vector>

#include "media/shot_cutter.h"
#include "render/grade_params.h"

namespace ajp::app::play {

inline constexpr std::int64_t kAnimeGlideUs = 250'000;  // le glissé d'un même plan mieux connu

// La médiane des APL des plans connus (nits) ; 0 sans plan.
double series_apl_of(const std::vector<media::Shot>& shots);

// Ce que la méthode tient d'un plan : sa lumière.
struct AnimeShotLook {
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
