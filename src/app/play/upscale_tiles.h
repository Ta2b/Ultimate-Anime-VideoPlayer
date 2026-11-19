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
// champ réceptif ; la sortie du cœur est celle d'un calcul complet. Ce qui peut dériver, c'est la sortie
// reprise : une tuile dite « inchangée » qui bouge un peu à chaque paire (nuage lent, fumée, fondu local).
// Trois gardes l'empêchent :
//   - par tuile, le mouvement reçu depuis son dernier calcul s'additionne (dx, dy de la carte, moins le
//     décalage appliqué) ; au-delà de kDriftMax pixels de travail, la tuile se recalcule ;
//   - par tuile, au plus kMaxAge images reprises de suite (comme la limite du lot A) ;
//   - toute l'image se recalcule au moins toutes les kFullEvery images, et à chaque coupe, fondu, image
//     abîmée, saut ou changement d'époque.
//
// Décalage : caméra ajustée, partie linéaire à l'identité (kernels::scene::kHeldLinear), résidu faible, et
// translation à moins de kShiftSnap d'un multiple du pas du moteur. La sortie d'avant se décale alors de ce
// multiple ; une tuile ne se reprend que si elle est du décor sûr qui suit ce décalage. La part fractionnaire
// s'accumule dans la dérive de chaque tuile.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "kernels/spec/scene_flow.h"
#include "kernels/spec/span_tiles.h"

namespace kirari::app::play {

// Ce que l'étage sait de l'image quand elle n'est pas reprise entière.
struct TileFacts {
  bool fresh = false;      // même époque, image saine, sortie de l'image d'avant présente et à jour
  bool can_shift = false;  // la sortie d'avant est dans un autre tampon que la nouvelle (décalage possible)
  const kernels::SceneMotion* scene = nullptr;  // la carte de (précédente, image) ; nulle : pas de carte
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
  static constexpr int kMaxAge = 24;           // images reprises de suite au plus, par tuile
  static constexpr int kFullEvery = 120;       // une image entière au moins toutes les 120 (5 s à 24 i/s)
  static constexpr float kDriftMax = 0.5F;     // pixels de travail reçus depuis le dernier calcul d'une tuile
  static constexpr double kShiftSnap = 0.125;  // écart au multiple du pas, pixels de travail
  static constexpr double kShiftResidual = 0.5;  // résidu de la caméra au plus, pixels de travail
  static constexpr float kFollow = 0.5F;         // écart d'une tuile décor au décalage, pixels de travail
  static constexpr float kSure = 0.5F;           // confiance d'une tuile décor reprise sous un décalage
  static constexpr double kMaxShare = 0.7;  // au-delà, la région dilatée coûte autant que l'image entière

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
  // « SPAN par tuiles : … » pour le journal de fin ; vide si rien n'a été fait par tuiles.
  std::string summary() const;

 private:
  struct Tile {
    int age = 0;
    float drift_x = 0.0F;  // mouvement reçu depuis le dernier calcul, pixels de travail
    float drift_y = 0.0F;
  };
  TilePlan full();
  // Le décalage (pixels d'entrée, multiples du pas) que la caméra permet ; (0, 0) sinon.
  bool camera_shift(const kernels::SceneMotion& scene, int& shift_x, int& shift_y) const;
  // Les tuiles de la carte sous la tuile (col, row) du moteur, rendues par appel de visit.
  template <class Visit>
  void scene_tiles_under(const kernels::SceneMotion& scene, int col, int row, Visit&& visit) const;

  kernels::SpanTileGrid grid_;
  std::vector<Tile> tiles_;
  int since_full_ = 0;
  bool known_ = false;  // l'état des tuiles suit la sortie d'avant
  // Bilan.
  std::uint64_t planned_ = 0;
  std::uint64_t tiled_ = 0;
  std::uint64_t shifted_ = 0;
  std::uint64_t tiles_done_ = 0;  // tuiles du cœur, sommées sur les images par tuiles
  std::uint64_t tiles_wide_ = 0;  // tuiles de la région dilatée
  std::uint64_t tiles_seen_ = 0;  // tuiles de la grille, sur les mêmes images
  std::uint64_t too_wide_ = 0;    // images entières parce que la région dépassait kMaxShare
  std::uint64_t no_map_ = 0;      // images entières faute de carte
  std::uint64_t refused_ = 0;
};

}  // namespace kirari::app::play
