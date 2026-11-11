// Noyaux CUDA de RIFE, sur des images RGB FP16 déjà ingérées à la taille de travail (libre, celle
// que la chaîne choisit) :
//   - l'entrée de RIFE (kernels/spec/rife.h) : 11 plans remplis au multiple de 64, image a,
//     image b, instant t, grilles et pas, noir de studio autour de la fenêtre ;
//   - le recadrage de la sortie remplie vers la fenêtre ;
//   - les mesures d'une paire (kernels/spec/detect.h) sur la luma tirée du RGB : somme des
//     écarts pour la coupe de scène, pire bloc 16 × 16 pour les dessins tenus (kernels/spec/detect.h).
// RIFE reçoit le RGB de la chaîne (matrice de la source) et sa sortie part telle quelle vers
// SPAN : pas d'aller-retour YUV, les deux réseaux voient le même espace.
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace ajp::gpu::cuda {

struct RifeInputParams {
  const void* a = nullptr;  // 1 × 3 × height × width, FP16
  const void* b = nullptr;
  int width = 0;  // fenêtre de l'image
  int height = 0;
  int padded_width = 0;  // rife_geometry
  int padded_height = 0;
  int left = 0;
  int top = 0;
  float black = 0.0F;     // noir de studio en RGB : 0 en plage limitée, 16/255 en plage pleine
  float time = 0.5F;      // instant t, 0 < t < 1
  void* input = nullptr;  // 1 × planes × padded_height × padded_width, FP16
  // 11 pour RIFE ; 7 pour RIFE v2 (images et instant, sans grilles) ; 6 : les deux images seules.
  int planes = 11;
  bool mesh = true;  // faux : le maillage (plans 7 à 10) déjà écrit reste tel quel (rife_pair)
};

cudaError_t launch_rife_input(const RifeInputParams& params, cudaStream_t stream);

struct RifeCropParams {
  const void* padded = nullptr;  // 1 × 3 × padded_height × padded_width, FP16
  int padded_width = 0;
  int padded_height = 0;
  int left = 0;
  int top = 0;
  int width = 0;
  int height = 0;
  void* out = nullptr;  // 1 × 3 × height × width, FP16
};

cudaError_t launch_rife_crop(const RifeCropParams& params, cudaStream_t stream);

// Mesures d'une paire : total = Σ |Ya − Yb| (Y en [0, 1], BT.709 sur le RGB), worst = pire bloc
// 16 × 16, moyenne des écarts × levels × 256 tronquée (1/256 de niveau 8 bits). Les deux sorties
// sont remises à zéro par l'appelant avant le lancement.
struct PairDiffParams {
  const void* a = nullptr;  // 1 × 3 × height × width, FP16
  const void* b = nullptr;
  int width = 0;
  int height = 0;
  float levels = 219.0F;  // pas de luma 8 bits entre noir et blanc : 219 limitée, 255 pleine
  // Coefficients de luma de la matrice de la source (kernels/spec/color.h) : la luma mesurée est celle du
  // flux (kernels/spec/detect.h), BT.601 pour un DVD, pas toujours BT.709 (B03-06).
  float kr = 0.2126F;
  float kb = 0.0722F;
  double* total = nullptr;
  unsigned int* worst = nullptr;
};

cudaError_t launch_pair_diff(const PairDiffParams& params, cudaStream_t stream);

}  // namespace ajp::gpu::cuda
