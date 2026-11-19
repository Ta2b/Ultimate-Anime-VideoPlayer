// Éclats HDR sur les seuls effets (render/effect_mask.h) : la carte de la scène de chaque paire (PairFacts::
// scene, kernels/spec/scene_flow.h), réduite par gpu à ce que le masque lit, notée ici avec l'intervalle de
// pts de la paire, puis lue par main à la présentation de chaque image (Session::show_request), comme les
// notes de pair_marks.h ; avec le grade avant le warp (source_grade.h), lue aussi par gpu pour chaque
// source gradée (EffectBook::for_source). Rien ne s'écrit sans carte (cartes autres que NVIDIA, NVOF refusé)
// ou avec
// --no-effect-highlights : l'écran garde alors le grade d'aujourd'hui.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>

#include "kernels/spec/scene_flow.h"
#include "render/effect_mask.h"

namespace kirari::app::play {

// Ce que le masque lit d'une carte : une valeur d'effet par tuile, les drapeaux de la paire, et la part de
// l'image que couvre la grille (work_width × work_height : l'image de travail de la carte).
//  - Effet : 1, quelle que soit la confiance (l'éclat d'une explosion ne doit pas attendre) ;
//  - Décor ou dessin : 1 − confiance (un verdict sûr éteint la marge, un verdict faible la garde) ;
//  - Incertain (aplat sans texture) : 0 s'il ne change pas (le ciel, la feuille blanche, l'intérieur d'un
//    aplat clair), sans verdict s'il change (le cœur plat d'un flash).
// Une paire tenue ou un fondu n'apprennent rien ; une coupe probable pose le masque d'un coup.
render::EffectTiles effect_tiles_of(const kernels::SceneMotion& scene, int work_width, int work_height,
                                    std::int64_t span_us, bool cut);

// Au-delà, la note la plus récente ne vaut plus pour une source gradée (EffectBook::for_source) : six images
// à 23,976 i/s.
inline constexpr std::int64_t kEffectStaleUs = 250'000;

struct EffectNote {
  std::uint32_t epoch = 0;
  std::int64_t from_us = 0;  // pts de a
  std::int64_t to_us = 0;    // pts de b
  std::shared_ptr<const render::EffectTiles> tiles;
};

// Les dernières paires, écrites par gpu et lues par main à chaque présentation (verrou court).
class EffectBook {
 public:
  // Assez pour l'avance de gpu sur l'écran (anneau et images en vol), comme PairMarks.
  static constexpr std::size_t kKept = 64;
  void add(EffectNote note);
  // La paire qui finit sur ou après pts_us et commence avant : l'image montrée en fait partie.
  std::optional<EffectNote> at(std::uint32_t epoch, std::int64_t pts_us) const;
  // Le grade avant le warp (source_grade.h) : la note la plus récente de l'époque, celle de la paire qui
  // finit sur la source en tête (ou la précédente, quand la source est agrandie en avance), si elle finit au
  // plus kEffectStaleUs avant pts_us. Plus ancienne, la carte s'est tue : rien, le masque revient au neutre.
  std::optional<EffectNote> for_source(std::uint32_t epoch, std::int64_t pts_us) const;
  void clear();

 private:
  mutable std::mutex mutex_;
  std::array<EffectNote, kKept> notes_{};
  std::size_t count_ = 0;
  std::size_t next_ = 0;
};

// Le masque lissé de l'écran (main, show), ou du grade avant le warp (gpu, source), nourri une fois par
// paire. Une image sans note (pas de carte, pas encore relue, autre époque) fait revenir le masque vers le
// neutre.
class EffectFollower {
 public:
  // L'image montrée (pts_us, kNoPts : inconnu) et la note de sa paire. Vrai si le masque a changé.
  bool show(const std::optional<EffectNote>& note, std::int64_t pts_us);
  // Une source gradée avant le warp, de l'époque epoch, et la note de EffectBook::for_source. Une autre
  // époque remet le masque à zéro ; sans note, il fait un pas d'une image vers le neutre. Vrai si le masque a
  // changé.
  bool source(std::uint32_t epoch, const std::optional<EffectNote>& note);
  void reset();
  const render::EffectMask& mask() const { return mask_; }

 private:
  render::EffectMask mask_;
  std::optional<std::uint32_t> epoch_;
  std::int64_t from_us_ = 0;
  std::int64_t to_us_ = 0;
  bool seen_ = false;
  std::optional<std::int64_t> last_pts_;  // la dernière image montrée sans note
};

}  // namespace kirari::app::play
