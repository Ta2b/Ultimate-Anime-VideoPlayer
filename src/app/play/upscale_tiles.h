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
// Exactitude (kernels/spec/span_tiles.h) : le moteur calcule toutes ses couches sur le cœur dilaté de son
// champ réceptif ; la sortie du cœur est celle d'un calcul complet. Le cœur lui-même est ce qui a changé
// dilaté de la même portée (une sortie voit l'entrée jusqu'à la portée), plus le rayon du débandage quand
// l'entrée est débandée. Ce qui peut dériver, c'est la sortie reprise : une tuile dite « inchangée » qui
// bouge ou s'éclaire un peu à chaque paire (nuage lent, fumée, lueur, fondu local, caméra lente). Cinq
// gardes :
//   - par tuile, le mouvement reçu depuis son dernier calcul s'additionne (dx, dy de la carte, moins le
//     décalage appliqué) ; au-delà de kDriftMax pixels de travail, la tuile se recalcule ;
//   - par tuile, le déplacement de la caméra reçu s'additionne (l'affine de la carte au coin de la tuile qui
//     bouge le plus, moins le décalage appliqué) ; au-delà de kCameraMax, le quart de pixel sous lequel la
//     carte dit la caméra tenue, la tuile se recalcule. Signé : le bruit de l'ajustement s'y annule, une
//     caméra lente s'y additionne ; un zoom ou une rotation ne gardent que les tuiles près de leur centre ;
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
// s'accumule dans la dérive de chaque tuile. Sans décalage, l'image n'est entière que si la caméra bouge
// d'au moins kCameraMax aux quatre coins et au centre (toute tuile serait à refaire). Les deux bords de l'axe
// décalé se recalculent : la bande qui entre n'a pas de source, et près du bord d'où le contenu sort, la
// sortie voit maintenant le bourrage de zéros de l'image.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
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
  // Déplacement de la caméra reçu depuis le dernier calcul d'une tuile, pixels de travail : la caméra tenue
  // de la carte.
  static constexpr auto kCameraMax = static_cast<float>(kernels::scene::kHeldCamera);
  static constexpr float kBiasMax = 1.0F;  // glissement de luminosité reçu, codes 8 bits (porte : 1 niveau)
  static constexpr double kShiftSnap = 0.125;    // écart au multiple du pas, pixels de travail
  static constexpr double kShiftResidual = 0.5;  // résidu de la caméra au plus, pixels de travail
  static constexpr float kFollow = 0.5F;         // écart d'une tuile décor au décalage, pixels de travail
  static constexpr float kSure = 0.5F;           // confiance d'une tuile décor reprise sous un décalage
  static constexpr double kMaxShare = 0.7;  // au-delà, la région dilatée coûte autant que l'image entière

  bool enabled = true;  // --no-span-tiles

  // La grille du moteur (gpu::Engine::span_tile_grid) ; inutilisable : jamais de tuiles. Remet l'état à zéro.
  void set_grid(const kernels::SpanTileGrid& grid);
  const kernels::SpanTileGrid& grid() const { return grid_; }
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
  // chaque raison des images entières (pour la région trop large, ce qui la remplissait), et ce que faisait
  // la caméra de la carte ; inactives, pourquoi (--no-span-tiles, moteur sans grille). Vide sans SPAN
  // (set_grid jamais appelé).
  std::string summary() const;

 private:
  struct Tile {
    int age = 0;
    float drift_x = 0.0F;  // mouvement reçu depuis le dernier calcul, pixels de travail
    float drift_y = 0.0F;
    float camera_x = 0.0F;  // déplacement de la caméra reçu depuis le dernier calcul, pixels de travail
    float camera_y = 0.0F;
    float bias = 0.0F;  // glissement de luminosité reçu depuis le dernier calcul, codes
  };
  // Pourquoi une tuile est à recalculer, la première raison dans cet ordre (bilan de la région trop large).
  enum class Cause : std::uint8_t { None, Border, Map, Age, Drift, Camera, Bias };
  static constexpr std::size_t kCauseCount = 7;
  // La caméra de la carte sur les images planifiées où elle est ajustée : tenue (sous kCameraMax aux quatre
  // coins et au centre), en partie (zoom, rotation, caméra entre les deux), ou qui bouge partout ; de
  // celles-ci, le déplacement au centre, par cases de kCameraBin pixel.
  static constexpr double kCameraBin = 0.05;
  struct CameraSeen {
    std::uint64_t held = 0;
    std::uint64_t partly = 0;
    std::uint64_t moving = 0;
    std::array<std::uint32_t, 81> center{};  // 80 cases jusqu'à 4 pixels, puis au-delà
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
  // Sous le décalage (sx, sy), la tuile n'a pas de source sûre : la bande qui entre (son contenu vient d'en
  // dehors de la grille) et les deux bords de l'axe décalé. À recalculer.
  bool unsourced(int col, int row, int sx, int sy) const;
  // Ce qui pousse à recalculer une tuile qui a reçu la paire (âge, dérive, caméra, luminosité), ou None.
  static Cause worn(const Tile& tile);
  // La caméra bouge d'au moins kCameraMax aux quatre coins de la grille et en son centre.
  bool camera_everywhere(const kernels::SceneMotion& scene) const;
  // Classe la caméra de l'image au bilan (CameraSeen).
  void note_camera(const kernels::SceneMotion& scene);
  // « région trop large (…) » : la part des tuiles en cause et de la région, et les causes.
  std::string wide_detail() const;
  // « caméra de la carte sur n images : … » ; vide sans caméra ajustée.
  std::string camera_detail() const;
  // Ce qui a changé (cœur brut) dilaté de la portée de la sortie : la portée du réseau, plus spread_px.
  std::vector<std::uint8_t> reach_of(const std::vector<std::uint8_t>& raw, int spread_px) const;
  // Le décalage (pixels d'entrée, multiples du pas) que la caméra permet ; (0, 0) sinon.
  bool camera_shift(const kernels::SceneMotion& scene, int& shift_x, int& shift_y) const;
  // Les tuiles de la carte sous la tuile (col, row) du moteur, rendues par appel de visit.
  template <class Visit>
  void scene_tiles_under(const kernels::SceneMotion& scene, int col, int row, Visit&& visit) const;

  kernels::SpanTileGrid grid_;
  bool span_ = false;  // set_grid appelé : l'étage a un SPAN
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
  std::array<std::uint64_t, kTileStaleCount> stale_{};    // par TileStale (None jamais), et l'état inconnu
  std::uint64_t recipe_ = 0;                              // l'entrée de SPAN a changé de recette
  std::uint64_t no_map_ = 0;                              // faute de carte
  std::uint64_t refresh_ = 0;                             // kFullEvery
  std::uint64_t cut_ = 0;                                 // coupe, fondu, carte sans tuiles
  std::uint64_t camera_ = 0;                              // caméra qui bouge partout sans décalage entier
  std::uint64_t too_wide_ = 0;                            // la région dépassait kMaxShare
  std::array<std::uint64_t, kCauseCount> wide_causes_{};  // ses tuiles en cause, par Cause
  std::uint64_t wide_region_ = 0;                         // ses tuiles de la région dilatée
  std::uint64_t wide_seen_ = 0;                           // ses tuiles de la grille
  CameraSeen camera_seen_;
  std::uint64_t refused_ = 0;  // le moteur a refusé le plan
  std::uint64_t checks_ = 0;
  std::uint32_t check_worst_ = 0;  // 1/256 de niveau
  double check_mean_ = 0.0;        // niveaux, le plus grand
};

}  // namespace kirari::app::play
