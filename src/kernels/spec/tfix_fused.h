// TemporalFix « fusionné » (lot B, étapes 2 et 3 ; docs/temporalfix.md) : le réseau de pifroggi coupé en
// cinq petits moteurs TensorRT (l'encodeur, une image ; les blocs 0 à 2 ; la tête de confiance), les six
// images d'appui en un lot de six, et tout le reste en noyaux CUDA (tfix_fused_kernels.cu) : flou 5 × 5 et
// pyramides, Resize, grilles et GridSample (la bibliothèque de warp), Concat, confiance et mélange final.
// Ce fichier donne les tailles du découpage et la disposition des tampons ; les noyaux et le moteur la
// lisent.
//
// Le graphe, relevé sur s1, s2 et s3 (référence : tools/temporalfix/fused_reference.py, 78 dB du graphe
// entier) :
//   - P_s(x) : x flouté (binomial 1 4 6 4 1 / 16, bords à zéro) puis réduit en bilinéaire demi-pixel à
//     ceil(H / s) × ceil(W / s), pour s = 8, 4, 2, sur l'image (3 canaux) et ses traits (4 canaux) ;
//   - bloc 0 : X0 = [P8(c), P8(k), traits P8(c), traits P8(k), t_k] (15 canaux) ;
//   - blocs 1 et 2 : le flot du niveau d'avant agrandi ×2 (valeurs doublées), P(k) déplacé par lui,
//     X = [P(c), P(k) déplacé, traits, traits déplacés, t_k, masque, 8 attributs, flot] (26 canaux) ;
//     chaque bloc rend 11 canaux (flot 2, masque 1, attributs 8) à 1/4 de sa résolution d'entrée, agrandis
//     à celle-ci ; son flot s'ajoute au flot d'entrée ;
//   - le flot du bloc 2 agrandi ×2 jusqu'à l'image, l'appui déplacé par lui (GridSample FP32, zéros dehors),
//     la tête de confiance à demi-résolution (25 canaux), puis le mélange temporel prudent.
// Le terme de coin du graphe (2^-11 · tanh(moyenne |appui|) ajouté au seul pixel (0, 0), moins d'un
// huitième de niveau 8 bits) n'est pas refait.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace kirari::kernels {

inline constexpr int kTfixFrames = 7;
inline constexpr int kTfixSupports = 6;
inline constexpr int kTfixCenter = 3;
inline constexpr int kTfixImage = 3;    // canaux de l'image
inline constexpr int kTfixTraits = 4;   // canaux des traits de l'encodeur
inline constexpr int kTfixPyramid = 7;  // image et traits, à chaque niveau
inline constexpr int kTfixX0 = 15;
inline constexpr int kTfixX = 26;
inline constexpr int kTfixY = 11;
inline constexpr int kTfixConf = 25;
inline constexpr int kTfixLevels = 3;  // niveaux 8, 4, 2

// L'image de chaque appui dans la fenêtre, et son temps (valeurs FP16 du graphe, onnx::Expand_182).
inline constexpr std::array<int, kTfixSupports> kTfixSupportFrame{0, 1, 2, 4, 5, 6};
inline constexpr std::array<float, kTfixSupports> kTfixTime{
    -1.0F, -0.66650390625F, -0.333251953125F, 0.333251953125F, 0.66650390625F, 1.0F};

constexpr int tfix_ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

// Le facteur du niveau l (0, 1, 2) : 8, 4, 2.
constexpr int tfix_level_scale(int level) {
  return 8 >> level;
}

struct TfixDims {
  int channels = 0;
  int height = 0;
  int width = 0;
  constexpr std::int64_t plane() const { return static_cast<std::int64_t>(height) * width; }
  constexpr std::int64_t elements() const { return channels * plane(); }
};

// Le niveau l d'une image du réseau height × width : ceil(H / s) × ceil(W / s).
constexpr TfixDims tfix_level(int level, int height, int width, int channels = kTfixPyramid) {
  const int s = tfix_level_scale(level);
  return {channels, tfix_ceil_div(height, s), tfix_ceil_div(width, s)};
}

// L'entrée du bloc b (au niveau b) et sa sortie avant son Resize : deux convolutions de pas 2.
constexpr TfixDims tfix_block_x(int block, int height, int width) {
  return tfix_level(block, height, width, block == 0 ? kTfixX0 : kTfixX);
}
constexpr TfixDims tfix_block_y(int block, int height, int width) {
  const TfixDims x = tfix_block_x(block, height, width);
  return {kTfixY, tfix_ceil_div(tfix_ceil_div(x.height, 2), 2), tfix_ceil_div(tfix_ceil_div(x.width, 2), 2)};
}

// La taille du réseau que les noyaux prennent : paire (le niveau 2 est alors exactement la moitié), au moins
// 16 pixels de côté.
constexpr bool tfix_size_valid(int height, int width) {
  return height >= 16 && width >= 16 && height % 2 == 0 && width % 2 == 0;
}

constexpr std::size_t tfix_align(std::size_t elements) {
  return (elements + 127U) / 128U * 128U;  // 256 octets en FP16
}

// Ce que l'encodeur fusionné écrit pour chaque image dans son tampon gardé (temporal_window, « traits ») :
// en demi-taille, l'image moyennée 2 × 2 (le réseau la lit) ; ses traits ; ses trois niveaux de pyramide
// (7 canaux chacun). Décalages en éléments FP16 depuis le début du tampon.
struct TfixPacked {
  std::size_t image = 0;  // demi-taille seulement
  std::size_t traits = 0;
  std::array<std::size_t, kTfixLevels> pyramid{};
  std::size_t elements = 0;
};

constexpr TfixPacked tfix_packed(int height, int width, bool half) {
  TfixPacked p;
  std::size_t at = 0;
  const auto take = [&at](std::int64_t n) {
    const std::size_t begin = at;
    at += tfix_align(static_cast<std::size_t>(n));
    return begin;
  };
  const TfixDims image{kTfixImage, height, width};
  if (half) p.image = take(image.elements());
  p.traits = take(TfixDims{kTfixTraits, height, width}.elements());
  for (int l = 0; l < kTfixLevels; ++l)
    p.pyramid[static_cast<std::size_t>(l)] = take(tfix_level(l, height, width).elements());
  p.elements = at;
  return p;
}

// Les plans du tampon gardé à la taille de travail (le réseau à demi-taille : la moitié de chaque côté).
constexpr int tfix_packed_planes(int net_height, int net_width, bool half) {
  const std::int64_t work = static_cast<std::int64_t>(net_height) * net_width * (half ? 4 : 1);
  return static_cast<int>(
      (static_cast<std::int64_t>(tfix_packed(net_height, net_width, half).elements) + work - 1) / work);
}

// Les tampons d'une fenêtre, dans la zone du moteur (éléments FP16, débuts alignés) : la table des adresses
// (TfixTable, 256 octets), X_b et Y_b des trois blocs, l'entrée de la tête, le masque moyenné, sa sortie, et
// en demi-taille la sortie du réseau avant d'être reportée sur l'image entière.
struct TfixLayout {
  std::size_t table = 0;
  std::array<std::size_t, kTfixLevels> x{};
  std::array<std::size_t, kTfixLevels> y{};
  std::size_t conf_in = 0;
  std::size_t mask_half = 0;
  std::size_t conf_out = 0;
  std::size_t net_out = 0;
  std::size_t elements = 0;
  constexpr std::size_t bytes() const { return elements * 2U; }
};

constexpr TfixLayout tfix_layout(int height, int width, bool half) {
  TfixLayout layout;
  std::size_t at = 0;
  const auto take = [&at](std::int64_t n) {
    const std::size_t begin = at;
    at += tfix_align(static_cast<std::size_t>(n));
    return begin;
  };
  layout.table = take(128);
  for (int b = 0; b < kTfixLevels; ++b) {
    const auto i = static_cast<std::size_t>(b);
    layout.x[i] = take(kTfixSupports * tfix_block_x(b, height, width).elements());
    layout.y[i] = take(kTfixSupports * tfix_block_y(b, height, width).elements());
  }
  const TfixDims half_level = tfix_level(2, height, width, 1);
  layout.conf_in = take(kTfixSupports * kTfixConf * half_level.plane());
  layout.mask_half = take(kTfixSupports * half_level.plane());
  layout.conf_out = take(kTfixSupports * half_level.plane());
  if (half) layout.net_out = take(TfixDims{kTfixImage, height, width}.elements());
  layout.elements = at;
  return layout;
}

}  // namespace kirari::kernels
