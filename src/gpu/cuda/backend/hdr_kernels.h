// Noyaux CUDA de RTX HDR (TrueHDR) dans la tranche verticale : l'image RGB FP16 de SPAN vers
// l'entrée 8 bits de Maxine, tramée avant la quantification (AH-392 : sans tramage, les quasi-noirs
// se crantent avant même TrueHDR), puis la sortie RGB10A2 (PQ BT.2020) vers le slot RGBA16 du ring.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace ajp::gpu::cuda {

struct ToRgba8Params {
  const void* rgb = nullptr;  // 1 × 3 × height × width, FP16, valeurs dans [0, 1]
  int width = 0;
  int height = 0;
  std::uint32_t seed = 0;  // change à chaque image : le bruit de tramage ne reste pas figé
  void* out = nullptr;     // R, G, B, A sur 8 bits par pixel, lignes de pitch octets
  std::size_t pitch = 0;
};

// Tramage triangulaire de ±1 code 8 bits (somme de deux bruits uniformes), puis quantification.
cudaError_t launch_to_rgba8(const ToRgba8Params& params, cudaStream_t stream);

struct Rgb10a2ToSurfaceParams {
  const void* in = nullptr;  // R bits 0-9, G 10-19, B 20-29, A 30-31, lignes de pitch octets
  std::size_t pitch = 0;
  int width = 0;
  int height = 0;
  cudaSurfaceObject_t surface = 0;  // RGBA16 UNORM : code 10 bits × 65535 / 1023, arrondi
};

cudaError_t launch_rgb10a2_to_surface(const Rgb10a2ToSurfaceParams& params, cudaStream_t stream);

inline constexpr int kLumaBins = 64;
inline constexpr int kLumaStatsFloats = 8;  // mean, p05, p50, p95, p995, dark, bright, numéro de mesure

struct MeanLumaParams {
  const void* rgb = nullptr;  // 1 × 3 × height × width, FP16, valeurs dans [0, 1] (comme ToRgba8Params)
  int width = 0;
  int height = 0;
  unsigned int* histogram = nullptr;  // kLumaBins compteurs sur le GPU, remis à 0 par la passe finale
  float* sum = nullptr;               // la somme des luminances sur le GPU, remise à 0 de même
  float* out = nullptr;  // kLumaStatsFloats flottants en mémoire hôte projetée : la dernière mesure
};

// Statistiques de luminance de l'image (0 à 1, pondération BT.709 des canaux tels que reçus), sur un point
// sur quatre dans chaque direction : histogramme de 64 cases par bloc puis global, moyenne, centiles 5, 50,
// 95 et 99,5 %, parts sous 10 % et au-dessus de 80 % ; deux lancements sur le flux, le second écrit le
// résultat en mémoire hôte sans copie (le numéro de mesure en dernier).
cudaError_t launch_mean_luma(const MeanLumaParams& params, cudaStream_t stream);

}  // namespace ajp::gpu::cuda
