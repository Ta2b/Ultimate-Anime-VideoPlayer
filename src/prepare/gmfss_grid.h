// Le GridNet de la fusion de GMFSS Fortuna Union, tel que les noyaux natifs le font (--gmfss-native,
// docs/gmfss-natif.md) : des tenseurs à quatre tailles et une liste de convolutions qui les lient, lue dans
// l'ONNX de la fusion par prepare/gmfss_native.h. En-tête sans std::expected : nvcc le lit en C++20.
#pragma once

#include <vector>

namespace kirari::prepare {

// Une convolution 3 × 3 de bord 1 (pas 1 ou 2), ou une transposée 4 × 4 de pas 2. Poids [cout][k²][cin]
// (k = 3, ou 4 pour la transposée, dont le poids ONNX [cin][cout][ky][kx] est rangé de même).
struct GmfssGridConv {
  bool transposed = false;
  int cin = 0;
  int cout = 0;
  int stride = 1;
  std::vector<float> weights;
  std::vector<float> bias;
  float in_slope = 1.0F;   // PReLU de l'entrée, appliqué à la lecture (1 : aucun)
  float out_slope = 1.0F;  // PReLU de la sortie, après l'addition résiduelle (1 : aucun)
};

// Un tenseur du réseau, rangé par pixel (NHWC FP16) : level 0 à la taille de la fusion h × w (la moitié des
// plans remplis), 1 et 2 à la moitié et au quart, kGmfssFullLevel au double (la sortie).
inline constexpr int kGmfssFullLevel = -1;
struct GmfssGridTensor {
  int level = 0;
  int channels = 0;  // canaux par pixel, multiple de 8
};

// Une convolution du réseau : elle lit from, ajoute skip (-1 : rien) et écrit to. shuffle : la convolution
// 64 → 256 suivie du PixelShuffle(2) de la queue, écrite par phases au double de la taille (le canal
// c · 4 + phase de la convolution est le canal c de la phase).
struct GmfssGridStep {
  GmfssGridConv conv;
  int from = 0;
  int skip = -1;
  int to = 0;
  bool shuffle = false;
};

// Les entrées : x0 (16 canaux : les trois plans de warped0_1, ceux de l'image de RIFE, ceux de warped1_1,
// puis sept nuls), x1 (les 64 caractéristiques de warped0_1 puis de warped1_1), x2 (warped0_2 puis
// warped1_2), x3 (warped0_3 puis warped1_3) ; la sortie : le dernier tenseur, 8 canaux au double de la taille
// dont les trois premiers sont l'image, à borner à [0, 1].
inline constexpr int kGmfssGridInputs = 4;
inline constexpr int kGmfssFeatureChannels = 64;  // les caractéristiques de warped*_1, après ses 3 plans
struct GmfssGrid {
  std::vector<GmfssGridTensor> tensors;
  std::vector<GmfssGridStep> steps;
};

// Les morceaux du GridNet, dont l'essai au chargement choisit les sommes à part (docs/gmfss-natif.md) : les
// lignes 0, 1 et 2 (le niveau du tenseur qu'écrit la convolution), puis la queue, de la convolution qui
// nourrit le PixelShuffle jusqu'à la fin. La coupe ne lit que la structure du plan, aucun nom de l'export.
inline constexpr int kGmfssQueuePiece = 3;
inline constexpr int kGmfssGridPieces = 4;
// Le morceau de chaque convolution ; vide si le réseau n'a pas cette forme (pas de PixelShuffle, rien avant
// lui, ou une convolution hors des trois lignes avant la queue).
std::vector<int> gmfss_grid_pieces(const GmfssGrid& grid);

}  // namespace kirari::prepare
