// Sortie YUV de l'export par puits CUDA (KernelOps::write_yuv ; contrat : kernels/ref rgb_to_yuv
// et kernels/spec/color.h) : un tenseur RGB planaire FP16 converti en NV12 ou P010 dans une image du
// pool de NVENC, sans passer par le ring ni par libplacebo.
//
// Deux passes. La première suit la ligne : chaque fil convertit les deux pixels d'une paire de
// colonnes, quantifie leur luma et réduit Cb et Cr en largeur par la table x (Spline36 vers la
// demi-largeur, au siting du flux) dans rows, en FP32. La seconde réduit rows en hauteur par la
// table y et quantifie la chroma entrelacée. Les pixels voisins que lit la réduction en largeur
// sont reconvertis par chaque fil, depuis le cache L1 : 12 prises de trois FP16 par échantillon.
#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

namespace ajp::gpu::cuda {

// Prises par échantillon au plus (Spline36 en réduction 2:1 : 12).
inline constexpr int kMaxYuvTaps = 16;

struct YuvWriteParams {
  const void* rgb = nullptr;  // 1 × 3 × height × width, FP16
  int width = 0;              // paires
  int height = 0;
  void* luma = nullptr;        // plan Y : width × height échantillons
  void* chroma = nullptr;      // plan CbCr entrelacé : (width / 2) × (height / 2) paires
  std::size_t luma_pitch = 0;  // octets
  std::size_t chroma_pitch = 0;
  bool ten_bit = false;  // P010 (codes de 10 bits << 6 sur 16 bits) ; NV12 sinon
  bool dither = false;   // seuil de Bayer 8 × 8 (quantize_dithered) ; arrondi au pair sinon
  // Matrice (coefficients) et normalisation du format (yuv_scale), quantification (quantizer).
  float kr = 0.2126F;
  float kb = 0.0722F;
  float yoff = 0.0F;
  float yscale = 1.0F;
  float coff = 0.0F;
  float cscale = 1.0F;
  float qdiv = 1.0F;
  float qmax = 255.0F;
  // Tables de ref::compute_weights en mémoire d'appareil : largeur → width / 2, hauteur → height / 2.
  const int* x_start = nullptr;
  const float* x_weights = nullptr;  // (width / 2) × x_taps
  int x_taps = 0;
  const int* y_start = nullptr;
  const float* y_weights = nullptr;  // (height / 2) × y_taps
  int y_taps = 0;
  float2* rows = nullptr;  // (width / 2) × height paires Cb, Cr réduites en largeur (yuv_rows_bytes)
};

std::size_t yuv_rows_bytes(int width, int height);

cudaError_t launch_write_yuv(const YuvWriteParams& params, cudaStream_t stream);

}  // namespace ajp::gpu::cuda
