// La reprise de SPAN sur les dessins tenus (lot A du chantier AnimeJaNai, chantier-animejanai/CHANTIER.md) :
// l'image arrivée reprend la sortie de SPAN de l'ancre, la dernière source vraiment agrandie, au lieu d'une
// nouvelle passe.
//
// L'idée vient de libaji (the-database/animejanai-inference, `src/core/held.{h,cpp}`, règle 2, et
// the-database/mpv@5b3f31d) : une ancre, une mesure ancre → image et une limite de reprises de suite. Rien
// n'en est recopié ; la règle ci-dessous est la nôtre, sur nos mesures (kernels/spec/detect.h) :
//
// - la première reprise d'une série lit la paire (précédente, image) : la précédente est l'ancre. Sans
//   interpolation aussi (alone) : la paire se mesure alors en avance, comme sous RIFE (look_ahead), et non
//   à l'arrivée de l'image, où la relecture attendait le SPAN de la précédente ;
// - les suivantes lisent l'écart entre l'ancre et l'image, sur les images de la mesure de la paire (la
//   copie réduite de RIFE quand elle existe, A09-18 ; l'image de travail sinon) : un fondu lent ou un
//   mouvement qui commence s'accumule contre l'ancre et passe le seuil de chaque bloc, là où des paires
//   voisines, chacune sous le seuil, laisseraient l'image se figer. C'est ce qui permet de passer de 2 à
//   kMaxInRow reprises de suite. Pas de critère sur l'écart moyen : un fondu change tous les blocs autant,
//   le pire bloc l'arrête donc aussi tôt ; l'écart moyen, lui, mesurait surtout le grain d'un BD (Spy×Family
//   13, deux de suite au plus) ;
// - sans mesure contre l'ancre (backend sans scores_now, allocation refusée), la règle d'avant : deux de
// suite ;
// - une image abîmée au décodage (DecodedFrame::corrupt) n'est ni reprise ni ancre.
//
// La décision rend une sorte (ReuseKind) et non un booléen. Quand l'image n'est pas reprise entière, les
// tuiles qui bougent (upscale_tiles.h, phase 3 du plan « flux optique ») peuvent ne recalculer que les tuiles
// changées, lues sur l'écart des blocs de la mesure de la paire : sorte Tiles, et le plan dans
// SpanReuse::tile_plan. Après une image par tuiles, la reprise entière demande aussi que chaque bloc reste au
// seuil (SpanTiles::holds) : la sortie y mêle des tuiles d'images différentes.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "app/play/upscale_tiles.h"
#include "gpu/backend.h"

namespace kirari::app::play {

// L'écart entre l'ancre et l'image arrivée (gpu::PairScores sur les images de la mesure de la paire).
struct AnchorGap {
  std::uint32_t worst_block = 0;  // pire bloc 16 × 16, en 1/256 de niveau 8 bits (kernels::is_held)
};

// Ce que l'étage sait de l'image arrivée.
struct ReuseFacts {
  bool same_epoch = false;          // ni début, ni saut
  bool corrupt = false;             // abîmée au décodage
  bool has_output = false;          // la sortie de SPAN de l'ancre est encore là
  bool pair_held = false;           // la paire (précédente, image) tenue ; vrai si seule l'ancre est mesurée
  std::optional<AnchorGap> anchor;  // mesure ancre → image ; absente au début d'une série mesurée par paire
  // Tuiles qui bougent (upscale_tiles.h). tiles_stale : ce qui empêche l'étage de les suivre pour cette image
  // (None : sortie d'avant à jour, ni TemporalFix, ni grade sur place sans copie, ni export, ni Smooth).
  TileStale tiles_stale = TileStale::Chain;
  int spread_px = 0;            // TileFacts::spread_px
  std::uint64_t input_key = 0;  // TileFacts::input_key
  PairBlocks blocks;            // TileFacts::blocks
};

// Ce que l'étage règle au démarrage (UpscaleSetup::span_reuse).
struct SpanReuseSetup {
  bool whole = true;   // --no-span-reuse : SPAN sur chaque image
  bool tiles = true;   // --no-span-tiles : jamais par tuiles, même quand l'image n'est pas reprise entière
  bool check = false;  // --span-tiles-check : une image par tuiles sur 24 comparée à un calcul complet
};

enum class ReuseKind : std::uint8_t {
  Compute,  // SPAN tourne, l'image devient l'ancre
  Whole,    // l'image entière reprend la sortie de l'ancre
  Tiles,    // SPAN ne recalcule que les tuiles de tile_plan ; l'image devient l'ancre, comme Compute
};

class SpanReuse {
 public:
  static constexpr int kMaxInRow = 24;      // comme libaji : un rafraîchissement au moins par seconde
  static constexpr int kPairOnlyInRow = 2;  // sans mesure contre l'ancre

  bool enabled = true;  // --no-span-reuse : SPAN sur chaque image
  bool alone = false;   // SPAN sans interpolation : l'étage mesure la paire pour la reprise, en avance
  // L'étage mesure-t-il chaque paire, à ×1 compris ? En SPAN seul, et pour les tuiles (l'écart des blocs).
  bool measures_pairs() const { return alone || tiles.active(); }
  // La mesure contre l'ancre sert-elle ? Pas au début d'une série dont la paire est mesurée (l'ancre est la
  // précédente).
  bool wants_anchor(bool pair_measured) const { return enabled && (in_row_ > 0 || !pair_measured); }
  int in_row() const { return in_row_; }
  ReuseKind decide(const ReuseFacts& facts);
  // Une reprise imposée par l'étage (copie d'une tenue fondue, Smooth) : comptée comme une reprise.
  void force();
  // Le pire bloc d'une paire mesurée de la même époque, au bilan : sous le seuil, juste au-dessus (du grain),
  // ou loin (un dessin qui change).
  void note_pair(std::uint32_t worst_block);
  // Le journal de fin : « SPAN repris : r sur n images (p %) … » ; vide si rien n'a été décidé.
  std::string summary() const;

  // L'image de l'ancre telle que la paire la mesure, copiée au début d'une série ; nul tant qu'aucune n'a
  // servi. Elle ne vaut que pour la série où elle a été copiée (anchor_kept) : une série imposée (force) ou
  // une image calculée la périme.
  std::unique_ptr<gpu::TensorBuffer> anchor;
  bool anchor_measure = true;                   // faux après un refus (backend sans scores_now, allocation)
  void anchor_kept() { anchor_ready_ = true; }  // l'étage vient de copier l'ancre de la série
  bool anchor_ready() const { return anchor_ready_; }

  // Les tuiles qui bougent, et le plan de l'image arrivée (tiled faux hors de ReuseKind::Tiles) : à donner au
  // moteur (gpu::Engine::next_span_tiles) juste avant le SPAN de cette image, puis à oublier.
  SpanTiles tiles;
  TilePlan tile_plan;
  bool check = false;                               // --span-tiles-check
  std::unique_ptr<gpu::TensorBuffer> check_output;  // le calcul complet du contrôle, à la taille de sortie
  // Avant le warp, le grade et la plaque de décor réécrivent le tampon de SPAN sur place : la sortie de SPAN
  // telle quelle est copiée juste avant le premier des deux (upscale_span.cpp, keep_ungraded), d'où la
  // suivante se calcule par tuiles. ungraded_of : le tampon de spanned_ dont c'est la copie ; nul s'il a été
  // réécrit depuis, ou si rien n'a été copié. plated : la plaque a réécrit ce tampon après son SPAN.
  std::unique_ptr<gpu::TensorBuffer> ungraded;
  std::optional<std::uint32_t> ungraded_of;
  bool ungraded_ok = true;  // faux après un refus de place ou de copie : plus de tuiles sous le grade
  std::array<bool, 2> plated{};
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
  bool anchor_bad_ = false;    // l'ancre est une image abîmée
  bool anchor_ready_ = false;  // anchor tient l'ancre de la série en cours
  std::uint64_t seen_ = 0;
  std::uint64_t reused_ = 0;
  std::uint64_t longest_ = 0;
  std::uint64_t stopped_limit_ = 0;
  std::uint64_t stopped_anchor_ = 0;  // paire tenue, mais l'écart à l'ancre passe le seuil
  std::uint64_t stopped_tiles_ = 0;   // tenue, mais un bloc d'une sortie par tuiles passe le seuil
  std::uint64_t corrupt_ = 0;
  // Les paires mesurées par pire bloc : jusqu'à 0,7, 1, 1,5, 2, 3 et 5 niveaux, puis au-delà (note_pair).
  std::array<std::uint64_t, 7> pairs_{};
};

}  // namespace kirari::app::play
