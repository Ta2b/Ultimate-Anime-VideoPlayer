// Noyaux CUDA de l'ingestion du backend TensorRT : NV12, P010 ou P016 (mémoire linéaire de NVDEC), ou YUV
// 4:2:0, 4:2:2 et 4:4:4 planaire du décodage logiciel envoyé par host_upload.h, vers RGB planaire FP16,
// réduction comprise quand le tenseur est plus petit que l'image (1080p vers 720p, une seule passe), et
// écriture RGB FP16 vers un slot RGBA16 UNORM du ring importé de Vulkan.
//
// La chroma suit le contrat de kernels/spec/resample.h, comme l'ingestion Vulkan et kernels/ref/color.cpp
// (B03-02) : suréchantillonnée en codes bruts, passe horizontale puis verticale, au filtre demandé
// (IngestSpec::chroma, Spline36 pour la chaîne), avec les décalages de chroma_shift_up et les bords en
// miroir ; la matrice s'applique ensuite point par point.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

#include "gpu/cuda/backend/ring_surface.h"

namespace ajp::gpu::cuda {

// Filtre de la réduction fusionnée (IngestParams::reduce_kind).
inline constexpr int kReduceArea = 0;
inline constexpr int kReduceTriangle = 1;
inline constexpr int kReduceCubic = 2;
inline constexpr int kReduceSpline36 = 3;
// Pixels de source par axe au plus pour un pixel de sortie (Spline36 : 6 × l'échelle + 1, soit toutes ses
// prises jusqu'à une réduction de 21 fois ; au-delà, retirées autant à gauche qu'à droite).
inline constexpr int kMaxReduceTaps = 128;

// Chroma de la source (IngestParams::chroma_layout).
inline constexpr int kChroma420 = 0;  // demi-largeur, demi-hauteur
inline constexpr int kChroma422 = 1;  // demi-largeur, pleine hauteur
inline constexpr int kChroma444 = 2;  // entière : lue au point, sans rééchantillonnage

struct IngestParams {
  const void* luma = nullptr;      // plan Y
  const void* chroma = nullptr;    // plan CbCr entrelacé, ou Cb seul en 4:2:0 planaire
  const void* chroma_v = nullptr;  // plan Cr en 4:2:0 planaire (décodage logiciel) ; nul : entrelacé
  std::size_t luma_pitch = 0;      // octets
  std::size_t chroma_pitch = 0;    // celui de Cb et de Cr en planaire
  int width = 0;                   // image décodée
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
  bool sixteen_bits = false;  // P010, P016, YUV 10 ou 16 bits : échantillons de 16 bits
  // Normalisation et matrice (kernels/spec/color.h : yuv_scale, coefficients).
  float yoff = 0.0F;
  float yscale = 1.0F;
  float coff = 0.0F;
  float cscale = 1.0F;
  float kr = 0.2126F;
  float kb = 0.0722F;
  // Chroma : disposition, taille de son plan, et tables du suréchantillonnage (kernels::ref::compute_weights,
  // décalage de chroma_shift_up compris) : pour la sortie i, x_taps poids x_weights[i × x_taps + j] des
  // points x_start[i] + j, passés au miroir (mirror_index). Sommes FP32 depuis 0, j croissant, comme la
  // référence. x_* en 4:2:0 et 4:2:2, y_* en 4:2:0 seulement.
  int chroma_layout = kChroma420;
  int chroma_width = 0;
  int chroma_height = 0;
  const int* x_start = nullptr;
  const float* x_weights = nullptr;
  int x_taps = 0;
  const int* y_start = nullptr;
  const float* y_weights = nullptr;
  int y_taps = 0;
  // 4:2:0 : la passe horizontale, faite une fois par ingest_chroma_rows avant les autres noyaux (width ×
  // chroma_height paires Cb, Cr en FP32, chroma_rows_bytes octets) ; la verticale se fait à la lecture.
  float* chroma_rows = nullptr;
  void* rgb = nullptr;  // 1 × 3 × height × width, FP16
  // Filtres étirés (triangle, cubique, Spline36) : deux passes séparables, la première convertit et
  // filtre les lignes dans rows (3 × height × out_width, FP32, ingest_rows_bytes octets), la
  // seconde filtre les colonnes. Les prises par pixel passent de nx × ny à nx + ny.
  // Depuis l'étape 5.3, une passe tuilée (ingest_tiled) fait les deux en mémoire partagée quand la
  // tuile tient dans 40 Ko ; rows ne sert plus qu'au-delà (réduction de plus de ~4 fois).
  float* rows = nullptr;
  int tile_rows = 0;
  // À la taille de l'image seulement : débande dans la même passe (étape 5.4, ingest_deband). Réduite,
  // l'image passe encore par launch_deband (recalculer la réduction aux 5 points coûterait ~5 passes).
  bool deband = false;  // posé par launch_ingest pour la passe tuilée (étape 5.3)
};

// Octets de IngestParams::rows pour ces paramètres ; 0 sans filtre séparable (même taille, Area).
std::size_t ingest_rows_bytes(const IngestParams& params);
// Octets de IngestParams::chroma_rows : 2 × width × chroma_height × 4 en 4:2:0, 0 sinon.
std::size_t chroma_rows_bytes(const IngestParams& params);

cudaError_t launch_ingest(const IngestParams& params, cudaStream_t stream);

struct WriteParams {
  const void* rgb = nullptr;  // 1 × 3 × height × width, FP16
  int width = 0;
  int height = 0;
  RingSurface surface{};  // emplacement de l'anneau (ring_surface.h) ; object 0 : aucun
};

cudaError_t launch_write_rgba16(const WriteParams& params, cudaStream_t stream);

}  // namespace ajp::gpu::cuda
