// Noyaux CUDA provisoires du backend TensorRT de la tranche verticale : ingestion NV12, P010 ou
// P016 (mémoire linéaire de NVDEC), ou YUV 4:2:0 planaire du décodage logiciel envoyé par
// host_upload.h, vers RGB planaire FP16, réduction Area comprise quand le tenseur est plus petit
// que l'image (1080p vers 720p, une seule passe), et écriture RGB FP16 vers un slot RGBA16 UNORM
// du ring importé de Vulkan.
//
// La chroma suit le contrat de la chaîne (IngestSpec::chroma, kChainChromaFilter) : Spline36 en codes
// bruts, par une passe horizontale dans un tampon (chroma_rows) puis une somme verticale de six prises à
// la conversion (kernels/spec/chroma_up.h, la formule que l'hôte et l'appareil partagent ; A03-01). Le
// bilinéaire d'avant reste pour IngestSpec::chroma = Bilinear et pour une image de dimension impaire : il
// suit la position de chroma de la source mais adoucit les bords de couleur, bien au-delà d'un niveau
// 8 bits sur un trait rouge ou des cheveux sur fond uni. Le socle CUDA annoncé (pc/cuda-compute) n'existe
// pas : ces noyaux sont ceux du lecteur.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

#include "gpu/cuda/backend/ring_surface.h"
#include "kernels/spec/chroma_up.h"

namespace ajp::gpu::cuda {

// Filtre de la réduction fusionnée (IngestParams::reduce_kind).
inline constexpr int kReduceArea = 0;
inline constexpr int kReduceTriangle = 1;
inline constexpr int kReduceCubic = 2;
inline constexpr int kReduceSpline36 = 3;
// Pixels de source par axe au plus pour un pixel de sortie (Spline36 : 6 × l'échelle + 1).
inline constexpr int kMaxReduceTaps = 32;

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
  bool sixteen_bits = false;  // P010, P016 ou YUV 4:2:0 10 bits : échantillons de 16 bits
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
  // Chroma Spline36 (kernels/spec/chroma_up.h) : les lignes de chroma filtrées vers la largeur de l'image
  // par launch_ingest (2 × ⌈height / 2⌉ × width, FP32, Cb puis Cr, ingest_chroma_bytes octets), et les deux
  // phases de chaque axe (chroma_up_phase, pour le décalage de la position de chroma). chroma_rows nul :
  // chroma bilinéaire, sans tampon.
  float* chroma_rows = nullptr;
  kernels::ChromaUpPhase chroma_wx[2];
  kernels::ChromaUpPhase chroma_wy[2];
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

// La chroma Spline36 se fait-elle pour cette image ? Dimensions paires (rapport 2 exact), au moins 2.
bool ingest_chroma_spline(int width, int height);
// Octets de IngestParams::chroma_rows pour cette image.
std::size_t ingest_chroma_bytes(int width, int height);

// Octets de IngestParams::rows pour ces paramètres ; 0 sans filtre séparable (même taille, Area).
std::size_t ingest_rows_bytes(const IngestParams& params);

cudaError_t launch_ingest(const IngestParams& params, cudaStream_t stream);

struct WriteParams {
  const void* rgb = nullptr;  // 1 × 3 × height × width, FP16
  int width = 0;
  int height = 0;
  RingSurface surface{};  // emplacement de l'anneau (ring_surface.h) ; object 0 : aucun
};

cudaError_t launch_write_rgba16(const WriteParams& params, cudaStream_t stream);

}  // namespace ajp::gpu::cuda
