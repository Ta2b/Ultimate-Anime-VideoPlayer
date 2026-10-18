// Contrat du rééchantillonnage séparable : noyaux, prises, positions, bords et ordre des passes.
//
// Repris de animejanai-inference-rtx@118c272 :
//   src/resample.h:31-98 (compute_filter de zimg, Spline36, bilinéaire, décalages de chroma),
//   src/kernels.cu:87-92 (mirr), 101-138 et 244-283 (passes et ordre d'accumulation),
//   src/pipeline_kernels.cu:11-50 (Area : intégration exacte de l'empreinte du pixel).
// Tests : tests/unit/kernels/resample_test.cpp.
#pragma once

#include "kernels/spec/color.h"

namespace ajp::kernels {

enum class Filter : std::uint8_t {
  Spline36,  // chaîne d'agrandissement et chroma de la chaîne (resize.Spline36)
  Bilinear,  // chroma de RIFE (resize.Bilinear de rife_cuda.py)
  Area,      // réduction seulement : moyenne exacte sur l'empreinte du pixel de sortie
};

// Dimensions d'une image de la chaîne : paires, dans [2, kMaxDimension]. Une dimension impaire
// ou hors plage est refusée. Les plans de chroma 4:2:0 (w/2, h/2) peuvent être impairs.
inline constexpr int kMaxDimension = 16384;

constexpr bool valid_frame_dimension(int n) {
  return n >= 2 && n <= kMaxDimension && n % 2 == 0;
}

constexpr bool valid_plane_dimension(int n) {
  return n >= 1 && n <= kMaxDimension;
}

// Noyau Spline36, support 3.
constexpr double spline36(double x) {
  x = x < 0.0 ? -x : x;
  if (x < 1.0) return ((13.0 / 11.0 * x - 453.0 / 209.0) * x - 3.0 / 209.0) * x + 1.0;
  if (x < 2.0) {
    x -= 1.0;
    return ((-6.0 / 11.0 * x + 270.0 / 209.0) * x - 156.0 / 209.0) * x;
  }
  if (x < 3.0) {
    x -= 2.0;
    return ((1.0 / 11.0 * x - 45.0 / 209.0) * x + 26.0 / 209.0) * x;
  }
  return 0.0;
}

// Noyau triangle, support 1.
constexpr double bilinear(double x) {
  x = x < 0.0 ? -x : x;
  return x < 1.0 ? 1.0 - x : 0.0;
}

// Support du noyau, en pixels source à l'échelle 1. Area n'a pas de noyau : son empreinte
// vaut src / dst pixels source.
constexpr double support(Filter f) {
  switch (f) {
    case Filter::Spline36:
      return 3.0;
    case Filter::Bilinear:
      return 1.0;
    case Filter::Area:
      return 0.0;
  }
  return 0.0;
}

// Table de poids, calculée en double puis rangée en float (compute_filter de zimg) :
//   scale = dst / src, step = min(scale, 1) : le noyau s'étire de 1/scale en réduction ;
//   taps = max(ceil(support / step) × 2, 1) ;
//   pos(i) = (i + 0,5) / scale + shift : centre de la sortie i dans la grille source ;
//   begin(i) = floor(pos − taps / 2 + 0,5) + 0,5 : centre de la première prise ;
//   start(i) = begin − 0,5 ; w(i, j) = f((begin + j − pos) × step), normalisés à une somme de 1.
// Area : empreinte [i × src / dst, (i + 1) × src / dst] ; le poids d'un pixel source est sa
// longueur de recouvrement divisée par celle de l'empreinte ; les prises manquantes valent 0.
// Bords : l'indice source passe par mirror_index avant lecture.
// Application : somme float, de j = 0 à taps − 1, depuis 0 ; passe horizontale, puis verticale ;
// intermédiaire float entre les passes, rien n'est borné avant la quantification finale.

// Miroir sans répétition du bord, puis clamp (kernels.cu mirr) : −1 → 0, n → n − 1.
constexpr int mirror_index(int i, int n) {
  i = i < 0 ? -i - 1 : i;
  i = i >= n ? 2 * n - 1 - i : i;
  if (i < 0) return 0;
  return i >= n ? n - 1 : i;
}

// Décalage de la chroma, en pixels du plan source, ajouté à pos(i) (convention zimg).
struct ChromaShift {
  double x = 0.0;
  double y = 0.0;
};

// 4:2:0 vers la grille de luma.
constexpr ChromaShift chroma_shift_up(Siting s) {
  return {.x = s == Siting::Center ? 0.0 : 0.25, .y = s == Siting::TopLeft ? 0.25 : 0.0};
}

// Grille de luma vers 4:2:0.
constexpr ChromaShift chroma_shift_down(Siting s) {
  return {.x = s == Siting::Center ? 0.0 : -0.5, .y = s == Siting::TopLeft ? -0.5 : 0.0};
}

// Ingestion 4:2:0 : la chroma brute (unités du conteneur) passe en horizontal puis en vertical,
// puis la matrice s'applique pixel par pixel (k_uv_h, k_v_f32, k_pre_combine).
// Sortie 4:2:0 : la matrice s'applique pixel par pixel, Y est quantifié aussitôt, Cb et Cr
// normalisés passent en horizontal puis en vertical, puis se quantifient (k_post_matrix,
// k_h_f32, k_uv_v_store). 4:4:4 : matrice seule, sans rééchantillonnage.
inline constexpr Filter kChainChromaFilter = Filter::Spline36;

}  // namespace ajp::kernels
