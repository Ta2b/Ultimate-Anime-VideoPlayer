// Descripteurs que les plans hôtes passent aux noyaux CUDA. Structures simples en C++20,
// partagées par les .cu et les .cpp : aucune vérification ici, les plans valident avant de
// lancer.
#pragma once

#include <cstddef>
#include <cstdint>

namespace ajp::gpu::cuda::dev {

// Taille d'un échantillon YUV dans sa mémoire.
enum class Sample : std::uint8_t { U8, U16 };

// Type des éléments d'un tenseur.
enum class Elem : std::uint8_t { F16, F32 };

// Un plan de pixels de l'appareil : adresse du premier octet, pas de ligne en octets.
struct Plane {
  std::uintptr_t ptr = 0;
  std::size_t pitch = 0;
};

// Tenseur planaire contigu (NCHW sans lot) : le plan c commence à c × width × height éléments.
// Une image RGB a 3 plans ; l'entrée de RIFE 11, le mouvement 4 et le masque 1.
struct Tensor {
  std::uintptr_t ptr = 0;
  Elem elem = Elem::F16;
  int width = 0;
  int height = 0;
  int channels = 3;
};

// Table de poids sur l'appareil (spec/resample.h) : dst sorties de taps prises chacune.
struct Weights {
  const int* start = nullptr;  // [dst] premier indice source, avant le miroir
  const float* w = nullptr;    // [dst × taps]
  int taps = 0;
  int src = 0;
  int dst = 0;
};

// Image YUV source, éventuellement placée dans une image remplie plus grande (RIFE) : hors de
// la source, la luma vaut pad_luma et la chroma pad_chroma.
struct YuvSource {
  Plane y;
  Plane u;  // CbCr entrelacés quand interleaved
  Plane v;
  Sample sample = Sample::U8;
  bool interleaved = true;
  bool subsampled = true;
  int width = 0;  // source, luma
  int height = 0;
  int left = 0;  // coin de la source dans l'image remplie, pair
  int top = 0;
  int padded_width = 0;  // image produite
  int padded_height = 0;
  std::uint16_t pad_luma = 0;
  std::uint16_t pad_chroma = 0;
};

// Destination YUV : la fenêtre (left, top, width, height) d'une image RGB plus grande, ou
// l'image entière.
struct YuvTarget {
  Plane y;
  Plane u;
  Plane v;
  Sample sample = Sample::U8;
  bool interleaved = true;
  bool subsampled = true;
  int left = 0;
  int top = 0;
  int width = 0;
  int height = 0;
};

// Normalisation et matrice de spec/color.h, déjà évaluées sur l'hôte.
struct YuvToRgb {
  float yoff = 0.0F;
  float yscale = 1.0F;
  float coff = 0.0F;
  float cscale = 1.0F;
  float kr = 0.0F;
  float kb = 0.0F;
};

struct RgbToYuv {
  YuvToRgb scale;
  float qdiv = 1.0F;  // quantizer() de spec/color.h
  float qmax = 255.0F;
};

// Plan de luma comparé par les détecteurs.
struct Luma {
  Plane plane;
  Sample sample = Sample::U8;
  int width = 0;
  int height = 0;
};

// Mouvement de RIFE : 4 plans de déplacements (f0x, f0y, f1x, f1y) puis le masque, de
// plane_width × plane_height, et la fenêtre de l'image dans ces plans.
struct Motion {
  std::uintptr_t flow = 0;
  std::uintptr_t mask = 0;
  Elem elem = Elem::F16;
  int plane_width = 0;
  int plane_height = 0;
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;
};

// Loi de mélange ajustée de spec/warp.h. Le terme de l'instant, B ln((1 − t) / t), est évalué
// sur l'hôte en float, comme dans la référence : le noyau n'a plus que le logit du masque.
struct BlendLaw {
  float a = 0.0F;
  float time_term = 0.0F;
  float c = 0.0F;
  float epsilon = 0.0F;
};

// Mesures d'une paire (spec/detect.h), relues d'un bloc par l'hôte.
struct PairResult {
  double score = 0.0;           // Σ d / aire remplie
  std::uint32_t worst = 0;      // pire bloc 32 × 32, en 1/256 de niveau 8 bits
  std::uint32_t identical = 0;  // 1 si la somme des écarts est nulle
};

// Courbes de spec/transfer.h : signal ↔ lumière linéaire (relative, ou cd/m² pour PQ).
enum class Curve : std::uint8_t { Linear, Srgb, Bt1886, Pq, Hlg };

// Une conversion par pixel : décodage vers la lumière linéaire, OOTF HLG éventuelle, gain,
// matrice de primaires éventuelle, puis encodage. Constantes déjà évaluées sur l'hôte.
struct TransferParams {
  Curve decode = Curve::Linear;
  Curve encode = Curve::Linear;
  bool hlg_ootf = false;
  bool convert = false;
  float hlg_peak = 1000.0F;
  float hlg_gamma = 1.2F;
  float hlg_kr = 0.0F;  // luma BT.2020 de la scène
  float hlg_kb = 0.0F;
  float bt1886_a = 1.0F;
  float bt1886_b = 0.0F;
  float bt1886_black = 0.0F;
  float bt1886_white = 1.0F;
  float gain = 1.0F;
  float matrix[9] = {1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F};
};

}  // namespace ajp::gpu::cuda::dev
