// Noyaux CUDA provisoires du backend TensorRT de la tranche verticale : ingestion NV12 ou
// P010 (mémoire linéaire de NVDEC) vers RGB planaire FP16, réduction Area comprise quand le
// tenseur est plus petit que l'image (1080p vers 720p, une seule passe), et écriture RGB FP16
// vers un slot RGBA16 UNORM du ring importé de Vulkan.
//
// Provisoire : le socle CUDA (chantier pc/cuda-compute, src/gpu/cuda/compute) porte les noyaux
// exacts de kernels/spec (chroma Spline36 en codes bruts). Ici la chroma est bilinéaire, avec
// la position de chroma de la source ; l'écart reste sous un niveau 8 bits sur les aplats.
// Ces noyaux disparaissent quand le socle est fusionné.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace ajp::gpu::cuda {

// Filtre de la réduction fusionnée (IngestParams::reduce_kind).
inline constexpr int kReduceArea = 0;
inline constexpr int kReduceTriangle = 1;
inline constexpr int kReduceCubic = 2;
inline constexpr int kReduceSpline36 = 3;
// Pixels de source par axe au plus pour un pixel de sortie (Spline36 : 6 × l'échelle + 1).
inline constexpr int kMaxReduceTaps = 32;

struct IngestParams {
  const void* luma = nullptr;    // plan Y
  const void* chroma = nullptr;  // plan CbCr entrelacé
  std::size_t luma_pitch = 0;    // octets
  std::size_t chroma_pitch = 0;
  int width = 0;  // image décodée
  int height = 0;
  // Tenseur de sortie : la taille de l'image, ou plus petit pour une réduction Area fusionnée
  // (chaque pixel de sortie moyenne sa boîte de source, pixels convertis en RGB un à un, pondérés
  // par leur recouvrement ; identique à convertir puis réduire, l'opération étant linéaire).
  int out_width = 0;
  int out_height = 0;
  // Filtre de la réduction : Area (boîte), triangle (bilinéaire étiré) ou cubique B, C
  // (Mitchell-Netravali), étirés à l'échelle de la réduction.
  int reduce_kind = kReduceArea;
  float reduce_b = 0.0F;
  float reduce_c = 0.0F;
  bool sixteen_bits = false;  // P010 ou P016 : échantillons de 16 bits
  // Normalisation et matrice (kernels/spec/color.h : yuv_scale, coefficients).
  float yoff = 0.0F;
  float yscale = 1.0F;
  float coff = 0.0F;
  float cscale = 1.0F;
  float kr = 0.2126F;
  float kb = 0.0722F;
  // Position de la chroma en pixels de luma : 0 pour la gauche, 0,5 au centre.
  float chroma_x = 0.0F;
  float chroma_y = 0.5F;
  void* rgb = nullptr;  // 1 × 3 × height × width, FP16
  // Filtres étirés (triangle, cubique, Spline36) : deux passes séparables, la première convertit et
  // filtre les lignes dans rows (3 × height × out_width, FP32, ingest_rows_bytes octets), la
  // seconde filtre les colonnes. Les prises par pixel passent de nx × ny à nx + ny.
  float* rows = nullptr;
};

// Octets de IngestParams::rows pour ces paramètres ; 0 sans filtre séparable (même taille, Area).
std::size_t ingest_rows_bytes(const IngestParams& params);

cudaError_t launch_ingest(const IngestParams& params, cudaStream_t stream);

struct WriteParams {
  const void* rgb = nullptr;  // 1 × 3 × height × width, FP16
  int width = 0;
  int height = 0;
  cudaSurfaceObject_t surface = 0;  // RGBA16 UNORM
};

cudaError_t launch_write_rgba16(const WriteParams& params, cudaStream_t stream);

}  // namespace ajp::gpu::cuda
