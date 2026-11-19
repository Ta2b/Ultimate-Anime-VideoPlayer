// Essai de tuiles_rejeu (--ancre-composee) : les tuiles de SPAN décidées sur les pixels, sans NVOF.
//
// Le lecteur jugeait une tuile inchangée sur la carte de la scène : vecteurs de NVOF et leur dérive cumulée,
// au budget de kStillDrift (0,025 pixel de travail). Il suit maintenant la variante summed ci-dessous
// (app/play/upscale_tiles.h, --span-tiles), à une différence près : la somme d'un bloc ne repart de zéro que
// quand toute sortie qui le voit a été recalculée, pas ses seules tuiles. Sur les extraits du PC
// (flux-carte, 19 novembre), la moyenne des vecteurs d'une tuile de la carte vaut 0,12 pixel de travail en
// médiane sur les plans à caméra tenue (0,55 au 90e centile) : le bruit de NVOF est cinq fois le budget, et
// la dérive y passe dès la première paire sur 90 % des tuiles.
//
// L'essai reprend la règle du lot A (app/play/upscale_reuse.h) tuile par tuile. L'ancre n'est plus une image
// mais l'image composée R de ce qui a produit la sortie : chaque tuile recalculée y recopie ses pixels de
// l'image. À chaque image, l'écart de chaque bloc de 16 pixels de travail entre l'image et R (la mesure de la
// paire, kernels/spec/detect.h) :
// - aucun bloc au-delà du seuil (0,70 niveau, kHeldThreshold) : la sortie se reprend entière, comme le lot A
//   (au plus SpanReuse::kMaxInRow de suite) ;
// - sinon, les tuiles d'un bloc au-delà du seuil (ou trop vieilles) se recalculent, dilatées de la
//   portée comme SpanTiles::plan, l'image entière si la région dépasse SpanTiles::kMaxShare ;
// - une coupe ou un fondu de la carte, ou kFullEvery images : l'image entière.
// Sans perte au sens du lot A : la sortie d'une tuile ne se reprend que si l'entrée qui l'a produite n'a
// bougé d'aucun bloc au-delà du seuil, glissement compris (un glissement de 0,025 pixel sur le pire bloc
// texturé en fait déjà 0,85 niveau), et un aplat qui glisse ne coûte rien.
//
// Variante sans image composée (summed) : l'écart de chaque bloc à ce qui a produit la sortie est majoré par
// la somme des écarts des paires voisines depuis le dernier calcul de sa tuile (l'écart moyen d'un bloc est
// une norme : inégalité triangulaire). Plus prudente (le grain s'additionne au lieu de rester à son niveau),
// elle ne demande au lecteur que la mesure de la paire (PairScores::blocks), sans tampon ni copie masquée.
//
// Variante bornée (anchored) : la même majoration D, resserrée par l'écart à la dernière image calculée A
// (la mesure contre l'ancre du lot A, par bloc) : D(image) = min(D(précédente) + paire, D(A) + écart à A),
// D(A) nul sur les tuiles calculées en A. Juste après une image entière, c'est la règle du lot A ; le grain
// ne s'additionne plus. Au lecteur : la mesure de la paire et celle contre l'ancre, par bloc, à chaque image.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "kernels/image.h"
#include "kernels/spec/scene_flow.h"
#include "kernels/spec/span_tiles.h"

namespace kirari::bench {

// L'écart de chaque bloc de side pixels entre deux lumas, en 1/256 de niveau, et le pire.
struct BlockGap {
  std::vector<std::uint32_t> deviation;
  int cols = 0;
  std::uint32_t worst = 0;
};
BlockGap block_gap(const kernels::Plane<std::uint8_t>& a, const kernels::Plane<std::uint8_t>& b, int side);

// Ce qui juge l'écart d'un bloc à l'entrée qui a produit sa sortie (en tête du fichier).
enum class Bound : std::uint8_t { Composed, Summed, Anchored };

class AnchorTiles {
 public:
  // grid : la grille du moteur ; divisor : pixels de travail par pixel des lumas ; side : côté d'un bloc de
  // la mesure, pixels des lumas ; spread_px : le rayon du débandage ; threshold : seuil d'un bloc, centièmes
  // de niveau (kernels::is_held).
  AnchorTiles(const kernels::SpanTileGrid& grid, int divisor, int side, int spread_px,
              std::uint32_t threshold, Bound bound, const kernels::Plane<std::uint8_t>& first);
  // L'image arrivée, et la carte de la paire (précédente, image) : 'R' reprise entière, 'T' par tuiles, 'C'
  // entière.
  char next(const kernels::Plane<std::uint8_t>& frame, const kernels::SceneMotion& scene);
  // Les tuiles que la dernière image a calculées (grid.count() octets, non nul : calculée).
  const std::vector<std::uint8_t>& last_core() const { return core_; }
  // « Ancre composée (essai) : … » pour la fin du rejeu.
  std::string summary() const;

 private:
  char full(const kernels::Plane<std::uint8_t>& frame, std::uint64_t& reason);
  std::vector<std::uint8_t> raw_of(const BlockGap& gap) const;
  // L'écart de chaque bloc à ce qui a produit la sortie : à R, ou sa majoration par les paires.
  BlockGap gap_of(const kernels::Plane<std::uint8_t>& frame);
  // La tuile i vient d'être calculée : R prend ses pixels, ses blocs repartent de zéro.
  void computed(std::size_t i, const kernels::Plane<std::uint8_t>& frame);

  kernels::SpanTileGrid grid_;
  int divisor_ = 1;
  int side_ = 16;
  int spread_px_ = 0;
  std::uint32_t threshold_ = 0;
  Bound bound_ = Bound::Composed;
  kernels::Plane<std::uint8_t> composed_;  // R
  kernels::Plane<std::uint8_t> previous_;  // l'image d'avant
  kernels::Plane<std::uint8_t> anchor_;    // A, la dernière image calculée
  std::vector<std::uint32_t> sums_;        // D, par bloc
  std::vector<std::uint32_t> frozen_;      // D(A)
  std::vector<int> age_;
  int since_full_ = 0;
  int in_row_ = 0;
  std::vector<std::uint8_t> core_;
  // Bilan.
  std::uint64_t seen_ = 0;
  std::uint64_t reused_ = 0;
  std::uint64_t tiled_ = 0;
  std::uint64_t tiles_core_ = 0;
  std::uint64_t tiles_wide_ = 0;
  std::uint64_t too_wide_ = 0;
  std::uint64_t cut_ = 0;
  std::uint64_t refresh_ = 0;
  std::uint64_t limit_ = 0;
};

}  // namespace kirari::bench
