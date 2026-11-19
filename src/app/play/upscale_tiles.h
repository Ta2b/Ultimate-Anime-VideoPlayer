// Les tuiles qui bougent (phase 3 du plan du flux optique NVIDIA, suite du lot A) : quand l'image n'est pas
// reprise entière (upscale_reuse.h), seules les tuiles qui ont changé repassent dans SPAN ; le reste reprend
// la sortie de l'image d'avant, décalée telle quelle sur un panoramique d'un nombre entier de pas.
//
// La décision se lit sur la carte de la scène de la paire (précédente, image) (kernels/spec/scene_flow.h) :
// son signal « a changé » vient du flux matériel de NVIDIA et ne prend ni le grain ni le bruit de compression
// pour du mouvement, là où une différence de pixels recalculerait ces tuiles pour rien. Sans carte (AMD,
// Intel, Turing, NVOF refusé, pas encore relue, début de lecture, saut), l'image entière se calcule, comme
// avant.
//
// Filet : la carte juge des tuiles de 32 × 32 pixels de travail (en 1080p) sur une image réduite de moitié,
// et ne dit « a changé » qu'à 6 codes d'écart moyen ou un vecteur net. Une petite bouche, un clignement dans
// un plan large y passent sous le seuil. L'écart de chaque bloc 16 × 16 de la mesure de la paire
// (gpu::PairScores::blocks, celui des dessins tenus) les voit : un bloc au-delà de kBlockChange recalcule
// les tuiles qu'il couvre. Le seuil est bien au-dessus du grain (un grain d'écart type σ donne un écart moyen
// d'environ 1,13 σ par bloc ; sur Spy×Family 13 BD, les paires sans mouvement restent sous un niveau au pire
// bloc), et bien en dessous d'une bouche (une cinquantaine de pixels qui changent de 80 codes : 15 niveaux).
// Sans portée sous un décalage (la mesure compare les mêmes pixels, pas le contenu suivi) : là, seul le décor
// sûr qui suit se reprend.
//
// Exactitude (kernels/spec/span_tiles.h) : le moteur calcule toutes ses couches sur le cœur dilaté de son
// champ réceptif ; la sortie du cœur est celle d'un calcul complet. Le cœur lui-même est ce qui a changé
// dilaté de la même portée (une sortie voit l'entrée jusqu'à la portée), plus le rayon du débandage quand
// l'entrée est débandée. Ce qui peut dériver, c'est la sortie reprise : une tuile dite « inchangée » qui
// bouge ou s'éclaire un peu à chaque paire (nuage lent, fumée, lueur, fondu local). Quatre gardes :
//   - par tuile, le mouvement reçu depuis son dernier calcul s'additionne (dx, dy de la carte, moins le
//     décalage appliqué) ; au-delà de kDriftMax pixels de travail, la tuile se recalcule ;
//   - par tuile, le glissement de luminosité reçu s'additionne (SceneTile::shift, où le grain s'annule) ;
//     au-delà de kBiasMax codes, la tuile se recalcule ;
//   - par tuile, au plus kMaxAge images reprises de suite (comme la limite du lot A) ;
//   - toute l'image se recalcule au moins toutes les kFullEvery images, à chaque coupe, fondu, image
//     abîmée, saut ou changement d'époque, et quand l'entrée de SPAN change de recette (débandage, filtre
//     de réduction).
//
// Décalage : caméra ajustée, partie linéaire à l'identité (kernels::scene::kHeldLinear), résidu faible, et
// translation à moins de kShiftSnap d'un multiple du pas du moteur. La sortie d'avant se décale alors de ce
// multiple ; une tuile ne se reprend que si elle est du décor sûr qui suit ce décalage. La part fractionnaire
// s'accumule dans la dérive de chaque tuile. Les deux bords de l'axe décalé se recalculent : la bande qui
// entre n'a pas de source, et près du bord d'où le contenu sort, la sortie voit maintenant le bourrage de
// zéros de l'image.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "kernels/spec/scene_flow.h"
#include "kernels/spec/span_tiles.h"

namespace kirari::app::play {

// Pourquoi la sortie de l'image d'avant ne se reprend pas par tuiles ; None : elle se reprend. Chaque image
// entière compte sous une raison au bilan (SpanTiles::summary).
enum class TileStale : std::uint8_t {
  None,
  Chain,        // la chaîne ne s'y prête pas : export, sortie YUV, TemporalFix, Smooth, plaque de décor
  Graded,       // le grade avant le warp l'a réécrite sur place, sans copie d'avant le grade
  Overwritten,  // ce n'est plus la sortie de la source d'avant : image créée par-dessus, source sautée
  Restart,      // début de lecture, saut, image ou ancre abîmée, pas de sortie d'avant
};
inline constexpr std::size_t kTileStaleCount = 5;

// L'écart de chaque bloc de la mesure de la paire (précédente, image), gpu::PairScores::blocks.
struct PairBlocks {
  std::span<const std::uint32_t> deviation;  // 1/256 de niveau, ligne par ligne ; vide : pas relu
  std::uint32_t cols = 0;
  float side_px = 0.0F;  // côté d'un bloc en pixels de travail (16 si la mesure est à la taille de travail)
};

// Ce que l'étage sait de l'image quand elle n'est pas reprise entière.
struct TileFacts {
  TileStale stale = TileStale::Restart;  // None : même époque, image saine, sortie d'avant présente et à jour
  bool can_shift = false;  // la sortie d'avant est dans un autre tampon que la nouvelle (décalage possible)
  const kernels::SceneMotion* scene = nullptr;  // la carte de (précédente, image) ; nulle : pas de carte
  // Ce qu'un pixel de la source touche de l'entrée de SPAN, au-delà de lui-même : le rayon du débandage,
  // pixels de travail ; 0 sans débandage.
  int spread_px = 0;
  // La recette de l'entrée de SPAN (débandage, filtre de réduction) : une autre que celle de la dernière
  // image calculée change tous les pixels un peu, sans que la carte le dise ; l'image est entière.
  std::uint64_t input_key = 0;
  PairBlocks blocks;  // le filet sous la carte ; vide : la carte seule
};

// Ce que l'image fera.
struct TilePlan {
  bool tiled = false;  // faux : l'image entière
  int shift_x = 0;     // pixels d'entrée, multiples de grid.shift_step
  int shift_y = 0;
  std::vector<std::uint8_t> core;  // grid.count() octets, non nul : à recalculer
};

class SpanTiles {
 public:
  static constexpr int kMaxAge = 24;        // images reprises de suite au plus, par tuile
  static constexpr int kFullEvery = 120;    // une image entière au moins toutes les 120 (5 s à 24 i/s)
  static constexpr float kDriftMax = 0.5F;  // pixels de travail reçus depuis le dernier calcul d'une tuile
  static constexpr float kBiasMax = 1.0F;   // glissement de luminosité reçu, codes 8 bits (porte : 1 niveau)
  static constexpr double kShiftSnap = 0.125;    // écart au multiple du pas, pixels de travail
  static constexpr double kShiftResidual = 0.5;  // résidu de la caméra au plus, pixels de travail
  static constexpr float kFollow = 0.5F;         // écart d'une tuile décor au décalage, pixels de travail
  static constexpr float kSure = 0.5F;           // confiance d'une tuile décor reprise sous un décalage
  static constexpr double kMaxShare = 0.7;  // au-delà, la région dilatée coûte autant que l'image entière
  static constexpr std::uint32_t kBlockChange = 200;  // écart d'un bloc qui change, centièmes de niveau

  bool enabled = true;  // --no-span-tiles

  // La grille du moteur (gpu::Engine::span_tile_grid) ; inutilisable : jamais de tuiles. Remet l'état à zéro.
  void set_grid(const kernels::SpanTileGrid& grid);
  bool active() const { return enabled && grid_.usable(); }
  // L'image se calcule : entière ou par tuiles. Met à jour l'état des tuiles comme si le plan était suivi.
  TilePlan plan(const TileFacts& facts);
  // L'image reprend la sortie d'avant entière (lot A) : chaque tuile vieillit et reçoit le mouvement de la
  // carte, s'il y en a une.
  void held(const kernels::SceneMotion* scene);
  // Le moteur a refusé les tuiles (next_span_tiles faux) : l'image est entière.
  void refused();
  // --span-tiles-check : vrai une image par tuiles sur kCheckEvery (la dernière planifiée), puis l'écart
  // mesuré entre sa sortie et un calcul complet (pire bloc 16 × 16 en 1/256 de niveau, écart moyen en
  // niveaux).
  static constexpr std::uint64_t kCheckEvery = 24;
  bool check_due() const { return tiled_ > 0 && tiled_ % kCheckEvery == 1; }
  void checked(std::uint32_t worst_block, double mean_levels);
  // « SPAN par tuiles : … » pour le journal de fin, même sans aucune image par tuiles : combien l'ont été, et
  // chaque raison des images entières. Vide si les tuiles ne sont pas actives (le démarrage l'a dit).
  std::string summary() const;

 private:
  struct Tile {
    int age = 0;
    float drift_x = 0.0F;  // mouvement reçu depuis le dernier calcul, pixels de travail
    float drift_y = 0.0F;
    float bias = 0.0F;  // glissement de luminosité reçu depuis le dernier calcul, codes
  };
  // Ce qu'une image par tuiles a ajouté au bilan, rendu si le moteur la refuse.
  struct Counted {
    std::uint64_t core = 0;
    std::uint64_t wide = 0;
    bool shifted = false;
  };
  TilePlan full();
  std::size_t at(int col, int row) const {
    return static_cast<std::size_t>(row) * static_cast<std::size_t>(grid_.cols) +
           static_cast<std::size_t>(col);
  }
  // L'image se calcule entière : le compteur de sa raison au bilan (sortie d'avant non reprise, état inconnu,
  // recette, pas de carte, rafraîchissement, coupe ou fondu, caméra qui bouge sans décalage entier) ; nul :
  // par tuiles si la région le permet.
  std::uint64_t* whole_reason(const TileFacts& facts, bool shifting, bool new_recipe);
  // L'état d'avant de la tuile (col, row), décalée de (sx, sy) : la pire des tuiles d'où vient son contenu.
  Tile carried(int col, int row, int sx, int sy) const;
  // Le contenu de la tuile vient d'en dehors de la grille (bande qui entre) : à recalculer.
  bool enters(int col, int row, int sx, int sy) const;
  // Ce qui a changé (cœur brut) dilaté de la portée de la sortie : la portée du réseau, plus spread_px.
  std::vector<std::uint8_t> reach_of(const std::vector<std::uint8_t>& raw, int spread_px) const;
  // Le décalage (pixels d'entrée, multiples du pas) que la caméra permet ; (0, 0) sinon.
  bool camera_shift(const kernels::SceneMotion& scene, int& shift_x, int& shift_y) const;
  // Les tuiles de la carte sous la tuile (col, row) du moteur, rendues par appel de visit.
  template <class Visit>
  void scene_tiles_under(const kernels::SceneMotion& scene, int col, int row, Visit&& visit) const;
  // Le filet : les tuiles sous un bloc qui change s'ajoutent à ce qui a changé (raw), hors décalage.
  void blocks_into(std::vector<std::uint8_t>& raw, const PairBlocks& blocks, bool shifting);

  kernels::SpanTileGrid grid_;
  std::vector<Tile> tiles_;
  int since_full_ = 0;
  bool known_ = false;           // l'état des tuiles suit la sortie d'avant
  std::uint64_t input_key_ = 0;  // la recette de l'entrée de la dernière image calculée
  Counted last_;                 // la dernière image par tuiles, pour refused()
  // Bilan.
  std::uint64_t planned_ = 0;
  std::uint64_t tiled_ = 0;
  std::uint64_t shifted_ = 0;
  std::uint64_t tiles_done_ = 0;  // tuiles du cœur, sommées sur les images par tuiles
  std::uint64_t tiles_wide_ = 0;  // tuiles de la région dilatée
  std::uint64_t tiles_seen_ = 0;  // tuiles de la grille, sur les mêmes images
  // Les images entières, chacune sous une seule raison.
  std::array<std::uint64_t, kTileStaleCount> stale_{};  // par TileStale (None jamais), et l'état inconnu
  std::uint64_t recipe_ = 0;                            // l'entrée de SPAN a changé de recette
  std::uint64_t no_map_ = 0;                            // faute de carte
  std::uint64_t refresh_ = 0;                           // kFullEvery
  std::uint64_t cut_ = 0;                               // coupe, fondu, carte sans tuiles
  std::uint64_t camera_ = 0;                            // caméra qui bouge sans décalage entier
  std::uint64_t too_wide_ = 0;                          // la région dépassait kMaxShare
  std::uint64_t refused_ = 0;                           // le moteur a refusé le plan
  std::uint64_t net_frames_ = 0;                        // images planifiées où le filet a vu ce que la carte taisait
  std::uint64_t net_tiles_ = 0;                         // et ses tuiles, avant dilatation
  std::uint64_t checks_ = 0;
  std::uint32_t check_worst_ = 0;  // 1/256 de niveau
  double check_mean_ = 0.0;        // niveaux, le plus grand
};

}  // namespace kirari::app::play
