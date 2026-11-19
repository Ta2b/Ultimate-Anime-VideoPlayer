// Les tuiles qui bougent (phase 3 du plan du flux optique NVIDIA, suite du lot A) : quand l'image n'est pas
// reprise entière (upscale_reuse.h), seules les tuiles qui ont changé repassent dans SPAN ; le reste reprend
// la sortie de l'image d'avant.
//
// La règle est celle du lot A, tuile par tuile : la sortie d'une tuile se reprend tant qu'aucun bloc de son
// entrée ne s'écarte de plus de kernels::kHeldThreshold (0,70 niveau) de l'entrée qui l'a produite. L'écart
// se lit sur la mesure de la paire (gpu::PairScores::blocks, l'écart moyen de chaque bloc 16 × 16 entre la
// précédente et l'image) : par bloc, la somme des écarts des paires depuis que toute sortie qui le voit a été
// recalculée majore son écart à l'entrée qui a produit chacune d'elles (l'écart moyen d'un bloc est une
// norme : inégalité triangulaire ; chaque paire compte 1/256 de niveau de plus, la troncature de la mesure).
// Un glissement sur un bord net s'y voit (0,025 pixel y fait déjà 0,85 niveau), un aplat qui glisse ne coûte
// rien ; le grain s'y additionne (la majoration est prudente). La somme se resserre par l'écart à l'ancre du
// lot A (la dernière image calculée), bloc par bloc : écart à l'ancre plus la majoration du bloc à l'ancre.
// Juste après une image entière, c'est la règle du lot A ; dans une série de reprises, le grain que l'ancre
// voit au même niveau ne s'additionne plus.
//
// Pourquoi pas la carte de la scène (NVOF) : sur les extraits du PC rejoués sur processeur
// (tools/bench/qualite/tuiles_rejeu, docs/tuiles-span.md), la moyenne des vecteurs d'une tuile vaut 0,12
// pixel de travail en médiane sur un plan à caméra tenue, cinq fois le glissement qu'une sortie reprise
// tolère (environ 34 niveaux par pixel au pire bloc, tools/bench/qualite/tuiles_panoramique.py) : aucune
// image n'y passait par tuiles. Un panoramique sous-pixel ne se reprend pas (le réseau ne commute pas avec un
// rééchantillonnage), et un panoramique d'un nombre entier de pas ne s'est vu sur aucun extrait : la sortie
// d'avant n'est plus décalée.
//
// Exactitude (kernels/spec/span_tiles.h) : le moteur calcule toutes ses couches sur le cœur dilaté de son
// champ réceptif ; la sortie du cœur est celle d'un calcul complet, comme toute sortie dont le champ tient
// dans la région calculée (elle s'écrit aussi). Le cœur est ce qui a changé dilaté de kCoreReachPx (la portée
// mesurée d'une entrée sur la sortie, sous la portée théorique du réseau), plus le rayon du débandage quand
// l'entrée est débandée. Gardes :
//   - par bloc, la somme des écarts des paires depuis le dernier calcul de toute sortie qui le voit, au seuil
//     du lot A ;
//   - par tuile, au plus kMaxAge images reprises de suite (comme la limite du lot A) ;
//   - toute l'image se recalcule au moins toutes les kFullEvery images, quand la région dépasse kMaxShare
//     (une coupe ou un fondu y passent), à chaque image abîmée, saut ou changement d'époque, sans mesure des
//     blocs ou quand sa géométrie change, et quand l'entrée de SPAN change de recette (débandage, filtre de
//     réduction) ;
//   - après une image par tuiles, la sortie mêle des tuiles d'images différentes : la reprise entière du lot
//     A n'a lieu que si chaque bloc reste aussi au seuil, aucune tuile au-delà de kMaxAge et l'image entière
//     pas encore due (holds).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "kernels/spec/detect.h"
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
  // Ce qu'un pixel de la source touche de l'entrée de SPAN, au-delà de lui-même : le rayon du débandage,
  // pixels de travail ; 0 sans débandage.
  int spread_px = 0;
  // La recette de l'entrée de SPAN (débandage, filtre de réduction) : une autre que celle de la dernière
  // image calculée change tous les pixels un peu, sans que la mesure de la paire le dise ; l'image est
  // entière.
  std::uint64_t input_key = 0;
  PairBlocks blocks;  // la mesure de la paire ; vide : l'image entière
  PairBlocks gap;     // l'écart de l'ancre (la dernière image calculée) à l'image, par bloc ; vide : inconnu
};

// Ce que l'image fera.
struct TilePlan {
  bool tiled = false;  // faux : l'image entière
  int shift_x = 0;     // décalage de la sortie d'avant (gpu::Engine::next_span_tiles) : toujours nul ici
  int shift_y = 0;
  std::vector<std::uint8_t> core;  // grid.count() octets, non nul : à recalculer
};

class SpanTiles {
 public:
  static constexpr int kMaxAge = 24;        // images reprises de suite au plus, par tuile
  static constexpr int kFullEvery = 120;    // une image entière au moins toutes les 120 (5 s à 24 i/s)
  static constexpr double kMaxShare = 0.7;  // au-delà, la région dilatée coûte autant que l'image entière
  // L'écart d'un bloc à l'entrée qui a produit sa sortie, au plus : le seuil des dessins tenus (lot A).
  static constexpr std::uint32_t kThreshold = kernels::kHeldThreshold;

  // La portée d'une entrée changée sur la sortie, en pixels de travail : au-delà, moins de 0,003 niveau
  // (SPAN Équilibre b8f64 sur processeur, carré de 32 pixels changé de 20 à 70 niveaux ; rien pour b5f48).
  // La portée du réseau (26 pixels de demi-résolution, quatre lignes de tuiles) reste celle de la région
  // calculée, que le moteur dilate lui-même.
  static constexpr int kCoreReachPx = 32;

  bool enabled = true;  // --no-span-tiles (UpscaleSetup::span_reuse)

  // La grille du moteur (gpu::Engine::span_tile_grid) ; inutilisable : jamais de tuiles. Remet l'état à zéro.
  void set_grid(const kernels::SpanTileGrid& grid);
  const kernels::SpanTileGrid& grid() const { return grid_; }
  bool active() const { return enabled && grid_.usable(); }
  // L'image se calcule : entière ou par tuiles. Met à jour l'état des tuiles comme si le plan était suivi.
  TilePlan plan(const TileFacts& facts);
  // La reprise entière du lot A garde-t-elle chaque bloc au seuil ? Toujours vrai tant que la sortie est
  // celle d'un calcul complet ; après une image par tuiles, chaque bloc avec la paire de l'image (ou l'écart
  // à l'ancre, gap), et ni kMaxAge ni kFullEvery dépassés.
  bool holds(const PairBlocks& blocks, const PairBlocks& gap) const;
  // L'image reprend la sortie d'avant entière (lot A) : chaque tuile vieillit et chaque bloc reçoit l'écart
  // de la paire, resserré par l'écart à l'ancre ; sans mesure, l'état ne suit plus (la prochaine image
  // calculée est entière).
  void held(const PairBlocks& blocks, const PairBlocks& gap);
  // Le moteur a refusé les tuiles (next_span_tiles faux) : l'image est entière.
  void refused();
  // --span-tiles-check : vrai une image par tuiles sur kCheckEvery (la dernière planifiée), puis l'écart
  // mesuré entre sa sortie et un calcul complet (pire bloc 16 × 16 en 1/256 de niveau, écart moyen en
  // niveaux).
  static constexpr std::uint64_t kCheckEvery = 24;
  bool check_due() const { return tiled_ > 0 && tiled_ % kCheckEvery == 1; }
  void checked(std::uint32_t worst_block, double mean_levels);
  // « SPAN par tuiles : … » pour le journal de fin, même sans aucune image par tuiles : combien l'ont été, et
  // chaque raison des images entières (pour la région trop large, ce qui la remplissait) ; demandées sur un
  // moteur sans grille, pourquoi. Vide sans SPAN (set_grid jamais appelé) ou avec --no-span-tiles.
  std::string summary() const;

 private:
  // Pourquoi une tuile est à recalculer, la première raison dans cet ordre (bilan de la région trop large).
  enum class Cause : std::uint8_t { None, Block, Age };
  static constexpr std::size_t kCauseCount = 3;
  // Ce qu'une image par tuiles a ajouté au bilan, rendu si le moteur la refuse.
  struct Counted {
    std::uint64_t core = 0;
    std::uint64_t wide = 0;
  };
  // L'image entière : l'état repart d'elle, à la géométrie de la mesure donnée.
  TilePlan full(const PairBlocks& blocks);
  // L'image se calcule entière : le compteur de sa raison au bilan ; nul : par tuiles si la région le permet.
  std::uint64_t* whole_reason(const TileFacts& facts, bool new_recipe);
  // La mesure a la géométrie des sommes (même nombre de blocs, même côté).
  bool fits(const PairBlocks& blocks) const;
  // La majoration du bloc b avec la paire de l'image, resserrée par l'écart à l'ancre quand il est mesuré.
  std::uint32_t bound(std::size_t b, const PairBlocks& blocks, const PairBlocks& gap) const;
  // Toutes les sommes nulles (la sortie est celle d'un calcul complet, première image comprise) : la
  // géométrie de la mesure les remplace, exactement. Faux sans mesure ou avec une somme non nulle.
  bool adopt(const PairBlocks& blocks);
  // Les tuiles de la grille sous le bloc b de la mesure, rendues par appel de visit.
  template <class Visit>
  void tiles_under(std::size_t b, Visit&& visit) const;
  // La portée de la sortie, en tuiles (colonnes, lignes) : kCoreReachPx (au plus celle du réseau), plus
  // spread_px.
  std::pair<int, int> reach(int spread_px) const;

  kernels::SpanTileGrid grid_;
  bool span_ = false;                // set_grid appelé : l'étage a un SPAN
  std::vector<int> age_;             // par tuile, images reprises depuis son dernier calcul
  std::vector<std::uint32_t> sums_;  // par bloc, la somme des écarts des paires (en tête du fichier)
  std::vector<std::uint32_t> base_;  // par bloc, sa majoration à l'ancre (la dernière image calculée)
  std::uint32_t block_cols_ = 0;     // la géométrie de la mesure des sommes
  float block_px_ = 0.0F;
  int since_full_ = 0;
  bool known_ = false;           // l'état des tuiles suit la sortie d'avant
  bool unmeasured_ = false;      // il ne la suit plus faute de mesure (held sans blocs) : au bilan
  bool mixed_ = false;           // la sortie mêle des tuiles calculées à des images différentes
  std::uint64_t input_key_ = 0;  // la recette de l'entrée de la dernière image calculée
  Counted last_;                 // la dernière image par tuiles, pour refused()
  // Bilan.
  std::uint64_t planned_ = 0;
  std::uint64_t tiled_ = 0;
  std::uint64_t tiles_done_ = 0;  // tuiles du cœur, sommées sur les images par tuiles
  std::uint64_t tiles_wide_ = 0;  // tuiles de la région dilatée
  std::uint64_t tiles_seen_ = 0;  // tuiles de la grille, sur les mêmes images
  // Les images entières, chacune sous une seule raison.
  std::array<std::uint64_t, kTileStaleCount> stale_{};    // par TileStale (None jamais), et l'état inconnu
  std::uint64_t recipe_ = 0;                              // l'entrée de SPAN a changé de recette
  std::uint64_t no_measure_ = 0;                          // sans mesure des blocs, ou d'une autre géométrie
  std::uint64_t refresh_ = 0;                             // kFullEvery
  std::uint64_t too_wide_ = 0;                            // la région dépassait kMaxShare
  std::array<std::uint64_t, kCauseCount> wide_causes_{};  // ses tuiles en cause, par Cause
  std::uint64_t wide_region_ = 0;                         // ses tuiles de la région dilatée
  std::uint64_t wide_seen_ = 0;                           // ses tuiles de la grille
  std::uint64_t refused_ = 0;                             // le moteur a refusé le plan
  std::uint64_t checks_ = 0;
  std::uint32_t check_worst_ = 0;  // 1/256 de niveau
  double check_mean_ = 0.0;        // niveaux, le plus grand
};

}  // namespace kirari::app::play
