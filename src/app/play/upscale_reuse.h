// La reprise de SPAN sur les dessins tenus (lot A du chantier AnimeJaNai, chantier-animejanai/CHANTIER.md) :
// l'image arrivée reprend la sortie de SPAN de l'ancre, la dernière source vraiment agrandie, au lieu d'une
// nouvelle passe.
//
// L'idée vient de libaji (the-database/animejanai-inference, `src/core/held.{h,cpp}`, règle 2, et
// the-database/mpv@5b3f31d) : une ancre, une mesure ancre → image et une limite de reprises de suite. Rien
// n'en est recopié ; la règle ci-dessous est la nôtre, sur nos mesures (kernels/spec/detect.h) :
//
// - la première reprise d'une série lit la paire (précédente, image) : la précédente est l'ancre ;
// - les suivantes lisent l'écart entre l'ancre et l'image, sur ce que SPAN lit (après TemporalFix et le
//   débandage) : un fondu lent ou un mouvement qui commence s'accumule contre l'ancre et passe le seuil, là
//   où des paires voisines, chacune sous le seuil, laisseraient l'image se figer. C'est ce qui permet de
//   passer de 2 à kMaxInRow reprises de suite ;
// - au-delà de kPairOnlyInRow, l'écart moyen contre l'ancre doit aussi rester sous kFadeMean (un changement
//   étendu, même faible partout, est un fondu) ;
// - sans mesure contre l'ancre (backend sans scores_now, allocation refusée), la règle d'avant : deux de
// suite ;
// - une image abîmée au décodage (DecodedFrame::corrupt) n'est ni reprise ni ancre.
//
// La décision rend une sorte (ReuseKind) et non un booléen. Quand l'image n'est pas reprise entière, les
// tuiles qui bougent (upscale_tiles.h, phase 3 du plan « flux optique ») peuvent ne recalculer que les tuiles
// changées, lues sur la carte de la scène : sorte Tiles, et le plan dans SpanReuse::tile_plan.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "app/play/upscale_tiles.h"
#include "gpu/backend.h"
#include "kernels/spec/scene_flow.h"

namespace kirari::app::play {

// L'écart entre l'ancre et l'image arrivée (gpu::PairScores sur les images que SPAN lit).
struct AnchorGap {
  std::uint32_t worst_block = 0;  // pire bloc 16 × 16, en 1/256 de niveau 8 bits (kernels::is_held)
  double mean_levels = 0.0;       // écart moyen de luma, en niveaux 8 bits
};

// Ce que l'étage sait de l'image arrivée.
struct ReuseFacts {
  bool same_epoch = false;          // ni début, ni saut
  bool corrupt = false;             // abîmée au décodage
  bool has_output = false;          // la sortie de SPAN de l'ancre est encore là
  bool pair_held = false;           // la paire (précédente, image) tenue ; vrai si seule l'ancre est mesurée
  std::optional<AnchorGap> anchor;  // mesure ancre → image ; absente au début d'une série mesurée par paire
  // Tuiles qui bougent (upscale_tiles.h). tiles_ok : l'étage sait les suivre pour cette image (sortie d'avant
  // à jour, ni TemporalFix, ni grade sur place, ni export, ni Smooth) ; can_shift : la sortie d'avant est
  // dans un autre tampon ; scene : la carte de la paire (précédente, image), nulle sans carte.
  bool tiles_ok = false;
  bool can_shift = false;
  const kernels::SceneMotion* scene = nullptr;
};

// Ce que l'étage règle au démarrage (UpscaleSetup::span_reuse).
struct SpanReuseSetup {
  bool whole = true;  // --no-span-reuse : SPAN sur chaque image
  bool tiles = true;  // --no-span-tiles : jamais par tuiles
};

enum class ReuseKind : std::uint8_t {
  Compute,  // SPAN tourne, l'image devient l'ancre
  Whole,    // l'image entière reprend la sortie de l'ancre
  Tiles,    // SPAN ne recalcule que les tuiles de tile_plan ; l'image devient l'ancre, comme Compute
};

class SpanReuse {
 public:
  static constexpr int kMaxInRow = 24;       // comme libaji : un rafraîchissement au moins par seconde
  static constexpr int kPairOnlyInRow = 2;   // sans mesure contre l'ancre, ou avant le critère de fondu
  static constexpr double kFadeMean = 0.25;  // niveaux 8 bits, au-delà de kPairOnlyInRow

  bool enabled = true;  // --no-span-reuse : SPAN sur chaque image
  // La mesure contre l'ancre sert-elle ? Pas au début d'une série dont la paire est mesurée (l'ancre est la
  // précédente).
  bool wants_anchor(bool pair_measured) const { return enabled && (in_row_ > 0 || !pair_measured); }
  int in_row() const { return in_row_; }
  ReuseKind decide(const ReuseFacts& facts);
  // Une reprise imposée par l'étage (copie d'une tenue fondue, Smooth) : comptée comme une reprise.
  void force();
  // Le journal de fin : « SPAN repris : r sur n images (p %) … » ; vide si rien n'a été décidé.
  std::string summary() const;

  // L'image de l'ancre telle que SPAN l'a lue, copiée au début d'une série ; nul tant qu'aucune n'a servi.
  std::unique_ptr<gpu::TensorBuffer> anchor;
  bool anchor_measure = true;  // faux après un refus (backend sans scores_now, allocation)

  // Les tuiles qui bougent, et le plan de l'image arrivée (tiled faux hors de ReuseKind::Tiles) : à donner au
  // moteur (gpu::Engine::next_span_tiles) juste avant le SPAN de cette image, puis à oublier.
  SpanTiles tiles;
  TilePlan tile_plan;
  // La source dont le tampon de SPAN de l'image (avant le warp) tient vraiment la sortie : posée par l'étage
  // à chaque SPAN fait ou repris. Une source sautée ne l'a pas : sa suivante ne se calcule pas par tuiles.
  std::optional<SourceId> spanned;
  // La source d'avant (l'étage la tient avec son identifiant id) est celle de spanned.
  template <class Previous>
  bool spanned_is(const std::optional<Previous>& previous) const {
    return previous.has_value() && spanned == previous->id;
  }

 private:
  void take();
  int in_row_ = 0;
  bool anchor_bad_ = false;  // l'ancre est une image abîmée
  std::uint64_t seen_ = 0;
  std::uint64_t reused_ = 0;
  std::uint64_t longest_ = 0;
  std::uint64_t stopped_limit_ = 0;
  std::uint64_t stopped_fade_ = 0;
  std::uint64_t corrupt_ = 0;
};

}  // namespace kirari::app::play
