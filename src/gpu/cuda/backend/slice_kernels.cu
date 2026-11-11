// Noyaux de l'ingestion du backend TensorRT (voir slice_kernels.h).
#include <cuda_fp16.h>

#include "gpu/cuda/backend/deband_kernels.cuh"
#include "gpu/cuda/backend/pixel_kernels.cuh"
#include "gpu/cuda/backend/slice_kernels.h"

namespace ajp::gpu::cuda {
namespace {

constexpr int kBlock = 16;

__device__ float sample(const void* plane, std::size_t pitch, int x, int y, bool sixteen_bits) {
  const auto* row = static_cast<const unsigned char*>(plane) + static_cast<std::size_t>(y) * pitch;
  if (sixteen_bits) return static_cast<float>(reinterpret_cast<const unsigned short*>(row)[x]);
  return static_cast<float>(row[x]);
}

__device__ int clamp_index(int i, int n) {
  return i < 0 ? 0 : (i >= n ? n - 1 : i);
}

// Miroir au demi-échantillon, puis bornage (kernels/spec/resample.h, mirror_index).
__device__ int mirror_index(int i, int n) {
  i = i < 0 ? -i - 1 : i;
  i = i >= n ? 2 * n - 1 - i : i;
  return i < 0 ? 0 : (i >= n ? n - 1 : i);
}

// Code brut de la chroma au point (x, y) de son plan ; c vaut 0 (Cb) ou 1 (Cr). Entrelacée : Cb et Cr
// alternent dans chroma. Planaire : Cb dans chroma, Cr dans chroma_v.
__device__ float chroma_code(const IngestParams& p, int x, int y, int c) {
  const bool planar = p.chroma_v != nullptr;
  const void* plane = planar && c == 1 ? p.chroma_v : p.chroma;
  return sample(plane, p.chroma_pitch, planar ? x : 2 * x + c, y, p.sixteen_bits);
}

// Passe horizontale de la chroma (resample_rows de la référence) : la colonne x de la luma, à la ligne cy
// du plan de chroma. Boucle non déroulée : les prises sont lues de la table.
__device__ float2 chroma_row(const IngestParams& p, int x, int cy) {
  const float* w = p.x_weights + static_cast<std::size_t>(x) * static_cast<std::size_t>(p.x_taps);
  const int first = p.x_start[x];
  float cb = 0.0F;
  float cr = 0.0F;
#pragma unroll 1
  for (int j = 0; j < p.x_taps; ++j) {
    const int cx = mirror_index(first + j, p.chroma_width);
    cb += w[j] * chroma_code(p, cx, cy, 0);
    cr += w[j] * chroma_code(p, cx, cy, 1);
  }
  return make_float2(cb, cr);
}

// Cb et Cr bruts au pixel (x, y) de la luma : lus au point en 4:4:4, passe horizontale à la volée en 4:2:2,
// passe verticale sur les lignes de ingest_chroma_rows en 4:2:0 (resample_columns de la référence).
__device__ float2 chroma_at(const IngestParams& p, int x, int y) {
  if (p.chroma_layout == kChroma444) return make_float2(chroma_code(p, x, y, 0), chroma_code(p, x, y, 1));
  if (p.chroma_layout == kChroma422) return chroma_row(p, x, y);
  const float* w = p.y_weights + static_cast<std::size_t>(y) * static_cast<std::size_t>(p.y_taps);
  const int first = p.y_start[y];
  const auto* rows = reinterpret_cast<const float2*>(p.chroma_rows);
  float cb = 0.0F;
  float cr = 0.0F;
#pragma unroll 1
  for (int j = 0; j < p.y_taps; ++j) {
    const std::size_t row = static_cast<std::size_t>(mirror_index(first + j, p.chroma_height));
    const float2 v = rows[row * static_cast<std::size_t>(p.width) + static_cast<std::size_t>(x)];
    cb += w[j] * v.x;
    cr += w[j] * v.y;
  }
  return make_float2(cb, cr);
}

// RGB du pixel (x, y) de l'image décodée : la chroma suréchantillonnée en codes bruts, puis normalisée et
// passée par la matrice (normalize et to_rgb de kernels/spec/color.h).
__device__ float3 rgb_at(const IngestParams& p, int x, int y) {
  const float luma = (sample(p.luma, p.luma_pitch, x, y, p.sixteen_bits) - p.yoff) / p.yscale;
  const float2 chroma = chroma_at(p, x, y);
  const float cb = (chroma.x - p.coff) / p.cscale;
  const float cr = (chroma.y - p.coff) / p.cscale;
  const float kg = 1.0F - p.kr - p.kb;
  return make_float3(luma + 2.0F * (1.0F - p.kr) * cr,
                     luma - (2.0F * p.kb * (1.0F - p.kb) * cb + 2.0F * p.kr * (1.0F - p.kr) * cr) / kg,
                     luma + 2.0F * (1.0F - p.kb) * cb);
}

// Passe horizontale de la chroma 4:2:0, une fois par image avant les autres noyaux : width × chroma_height
// paires Cb, Cr en FP32 dans chroma_rows.
__global__ void __launch_bounds__(kPixelThreads, kPixelBlocksPerSm) ingest_chroma_rows(IngestParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int cy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || cy >= p.chroma_height) return;
  const std::size_t at =
      static_cast<std::size_t>(cy) * static_cast<std::size_t>(p.width) + static_cast<std::size_t>(x);
  reinterpret_cast<float2*>(p.chroma_rows)[at] = chroma_row(p, x, cy);
}

// Moyenne Area du pixel de sortie (x, y) : la boîte [x·sx, (x+1)·sx) × [y·sy, (y+1)·sy) de
// l'image, chaque pixel de source pondéré par son recouvrement (3 × 3 pixels au plus à 1,5).
__device__ float3 area_at(const IngestParams& p, int x, int y) {
  const float sx = static_cast<float>(p.width) / static_cast<float>(p.out_width);
  const float sy = static_cast<float>(p.height) / static_cast<float>(p.out_height);
  const float x0 = static_cast<float>(x) * sx;
  const float y0 = static_cast<float>(y) * sy;
  const float x1 = x0 + sx;
  const float y1 = y0 + sy;
  const int last_x = min(static_cast<int>(ceilf(x1)), p.width);
  const int last_y = min(static_cast<int>(ceilf(y1)), p.height);
  float3 sum = make_float3(0.0F, 0.0F, 0.0F);
#pragma unroll 1
  for (int j = static_cast<int>(floorf(y0)); j < last_y; ++j) {
    const float wy = fminf(y1, static_cast<float>(j + 1)) - fmaxf(y0, static_cast<float>(j));
#pragma unroll 1
    for (int i = static_cast<int>(floorf(x0)); i < last_x; ++i) {
      const float w = wy * (fminf(x1, static_cast<float>(i + 1)) - fmaxf(x0, static_cast<float>(i)));
      const float3 c = rgb_at(p, i, j);
      sum.x += c.x * w;
      sum.y += c.y * w;
      sum.z += c.z * w;
    }
  }
  const float norm = 1.0F / (sx * sy);
  return make_float3(sum.x * norm, sum.y * norm, sum.z * norm);
}

// Noyau de la réduction à la distance d (en pixels de sortie) : triangle (rayon 1) ou cubique de
// Mitchell-Netravali B, C (rayon 2).
__device__ float reduce_weight(const IngestParams& p, float d) {
  d = fabsf(d);
  if (p.reduce_kind == kReduceTriangle) return fmaxf(0.0F, 1.0F - d);
  if (p.reduce_kind == kReduceSpline36) {
    if (d < 1.0F) return ((13.0F / 11.0F * d - 453.0F / 209.0F) * d - 3.0F / 209.0F) * d + 1.0F;
    if (d < 2.0F) {
      d -= 1.0F;
      return ((-6.0F / 11.0F * d + 270.0F / 209.0F) * d - 156.0F / 209.0F) * d;
    }
    if (d < 3.0F) {
      d -= 2.0F;
      return ((1.0F / 11.0F * d - 45.0F / 209.0F) * d + 26.0F / 209.0F) * d;
    }
    return 0.0F;
  }
  const float b = p.reduce_b;
  const float c = p.reduce_c;
  if (d < 1.0F) {
    return ((12.0F - 9.0F * b - 6.0F * c) * d * d * d + (-18.0F + 12.0F * b + 6.0F * c) * d * d +
            (6.0F - 2.0F * b)) *
           (1.0F / 6.0F);
  }
  if (d < 2.0F) {
    return ((-b - 6.0F * c) * d * d * d + (6.0F * b + 30.0F * c) * d * d + (-12.0F * b - 48.0F * c) * d +
            (8.0F * b + 24.0F * c)) *
           (1.0F / 6.0F);
  }
  return 0.0F;
}

// Pixel (x, y) du tenseur de sortie, en FP16.
__device__ void store_rgb(const IngestParams& p, int x, int y, float3 c) {
  const std::size_t plane = static_cast<std::size_t>(p.out_width) * static_cast<std::size_t>(p.out_height);
  const std::size_t i =
      static_cast<std::size_t>(y) * static_cast<std::size_t>(p.out_width) + static_cast<std::size_t>(x);
  auto* out = static_cast<__half*>(p.rgb);
  out[i] = __float2half_rn(c.x);
  out[plane + i] = __float2half_rn(c.y);
  out[2 * plane + i] = __float2half_rn(c.z);
}

// Prises du filtre étiré pour le pixel de sortie o d'un axe (in pixels de source, out de sortie) :
// chaque pixel de source à moins du rayon × échelle du centre, comme la réduction de mpv.
struct Taps {
  int first = 0;
  int count = 0;
  float center = 0.0F;
  float scale = 1.0F;
};

__device__ Taps taps_for(const IngestParams& p, int o, int in, int out) {
  const float scale = static_cast<float>(in) / static_cast<float>(out);
  const float radius = p.reduce_kind == kReduceTriangle   ? 1.0F
                       : p.reduce_kind == kReduceSpline36 ? 3.0F
                                                          : 2.0F;
  const float center = (static_cast<float>(o) + 0.5F) * scale - 0.5F;
  int first = static_cast<int>(ceilf(center - radius * scale));
  int count = static_cast<int>(floorf(center + radius * scale)) - first + 1;
  // Au-delà du plafond (réduction de plus de 21 fois en Spline36), autant de prises en moins de chaque côté :
  // le noyau reste centré, l'image ne se décale pas (B03-07).
  if (count > kMaxReduceTaps) {
    first += (count - kMaxReduceTaps) / 2;
    count = kMaxReduceTaps;
  }
  return Taps{.first = first, .count = count, .center = center, .scale = scale};
}

__device__ float tap_weight(const IngestParams& p, const Taps& t, int i) {
  return reduce_weight(p, (static_cast<float>(t.first + i) - t.center) / t.scale);
}

// Première passe des filtres étirés : la ligne y de la source, convertie en RGB et filtrée vers
// out_width colonnes, poids normalisés, bords répétés. Boucles non déroulées : sm_120 déroulerait
// sinon les prises et déborderait des 64 registres.
__global__ void __launch_bounds__(kPixelThreads, kPixelBlocksPerSm) ingest_rows(IngestParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.out_width || y >= p.height) return;
  const Taps t = taps_for(p, x, p.width, p.out_width);
  float3 line = make_float3(0.0F, 0.0F, 0.0F);
  float total = 0.0F;
#pragma unroll 1
  for (int i = 0; i < t.count; ++i) {
    const float w = tap_weight(p, t, i);
    total += w;
    if (w == 0.0F) continue;
    const float3 c = rgb_at(p, clamp_index(t.first + i, p.width), y);
    line.x += c.x * w;
    line.y += c.y * w;
    line.z += c.z * w;
  }
  const float norm = 1.0F / total;
  const std::size_t plane = static_cast<std::size_t>(p.out_width) * static_cast<std::size_t>(p.height);
  const std::size_t i =
      static_cast<std::size_t>(y) * static_cast<std::size_t>(p.out_width) + static_cast<std::size_t>(x);
  p.rows[i] = line.x * norm;
  p.rows[plane + i] = line.y * norm;
  p.rows[2 * plane + i] = line.z * norm;
}

// Seconde passe : la colonne x des lignes filtrées, filtrée vers out_height lignes.
__global__ void __launch_bounds__(kPixelThreads, kPixelBlocksPerSm) ingest_columns(IngestParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.out_width || y >= p.out_height) return;
  const Taps t = taps_for(p, y, p.height, p.out_height);
  const std::size_t in_plane = static_cast<std::size_t>(p.out_width) * static_cast<std::size_t>(p.height);
  float3 sum = make_float3(0.0F, 0.0F, 0.0F);
  float total = 0.0F;
#pragma unroll 1
  for (int j = 0; j < t.count; ++j) {
    const float w = tap_weight(p, t, j);
    if (w == 0.0F) continue;
    total += w;
    const float* row =
        p.rows +
        static_cast<std::size_t>(clamp_index(t.first + j, p.height)) * static_cast<std::size_t>(p.out_width) +
        x;
    sum.x += row[0] * w;
    sum.y += row[in_plane] * w;
    sum.z += row[2 * in_plane] * w;
  }
  const float norm = 1.0F / total;
  store_rgb(p, x, y, make_float3(sum.x * norm, sum.y * norm, sum.z * norm));
}

// Réduction étirée en une passe tuilée (étape 5.3) : un bloc de 32 × 8 fils rend une tuile de sortie de
// kTileWidth × kTileHeight. Les lignes de source dont la tuile a besoin sont d'abord filtrées vers ses 32
// colonnes en mémoire partagée (FP32), puis chaque fil en tire deux pixels de sortie. Mêmes prises, mêmes
// sommes dans le même ordre et même normalisation que ingest_rows puis ingest_columns : même résultat, sans
// le tampon rows ni son aller-retour en mémoire globale.
constexpr int kTileWidth = 32;
constexpr int kTileHeight = 16;
constexpr int kTileThreadsY = 8;

__global__ void __launch_bounds__(kTileWidth* kTileThreadsY) ingest_tiled(IngestParams p) {
  extern __shared__ float tile[];  // 3 plans de p.tile_rows × kTileWidth
  const int x0 = static_cast<int>(blockIdx.x) * kTileWidth;
  const int y0 = static_cast<int>(blockIdx.y) * kTileHeight;
  const int y_last = min(y0 + kTileHeight, p.out_height) - 1;
  const Taps top = taps_for(p, y0, p.height, p.out_height);
  const Taps bottom = taps_for(p, y_last, p.height, p.out_height);
  const int first_row = top.first;
  const int rows = min(bottom.first + bottom.count - first_row, p.tile_rows);
  const std::size_t tile_plane = static_cast<std::size_t>(p.tile_rows) * kTileWidth;
  const int thread = static_cast<int>(threadIdx.y) * kTileWidth + static_cast<int>(threadIdx.x);

  // 1. Les lignes first_row… de la source (bords répétés), filtrées vers les colonnes x0… de la tuile.
#pragma unroll 1
  for (int k = thread; k < rows * kTileWidth; k += kTileWidth * kTileThreadsY) {
    const int r = k / kTileWidth;
    const int cx = k - r * kTileWidth;
    const int x = x0 + cx;
    if (x >= p.out_width) continue;
    const int y = clamp_index(first_row + r, p.height);
    const Taps t = taps_for(p, x, p.width, p.out_width);
    float3 line = make_float3(0.0F, 0.0F, 0.0F);
    float total = 0.0F;
#pragma unroll 1
    for (int i = 0; i < t.count; ++i) {
      const float w = tap_weight(p, t, i);
      total += w;
      if (w == 0.0F) continue;
      const float3 c = rgb_at(p, clamp_index(t.first + i, p.width), y);
      line.x += c.x * w;
      line.y += c.y * w;
      line.z += c.z * w;
    }
    const float norm = 1.0F / total;
    tile[k] = line.x * norm;
    tile[tile_plane + k] = line.y * norm;
    tile[2 * tile_plane + k] = line.z * norm;
  }
  __syncthreads();

  // 2. Chaque fil : la colonne x0 + threadIdx.x, aux lignes de sortie y0 + threadIdx.y et + 8.
  const int x = x0 + static_cast<int>(threadIdx.x);
  if (x >= p.out_width) return;
#pragma unroll 1
  for (int y = y0 + static_cast<int>(threadIdx.y); y <= y_last; y += kTileThreadsY) {
    const Taps t = taps_for(p, y, p.height, p.out_height);
    float3 sum = make_float3(0.0F, 0.0F, 0.0F);
    float total = 0.0F;
#pragma unroll 1
    for (int j = 0; j < t.count; ++j) {
      const float w = tap_weight(p, t, j);
      if (w == 0.0F) continue;
      total += w;
      const int r = t.first + j - first_row;
      if (r < 0 || r >= rows) continue;  // jamais : la tuile couvre les prises de ses lignes
      const std::size_t at = static_cast<std::size_t>(r) * kTileWidth + threadIdx.x;
      sum.x += tile[at] * w;
      sum.y += tile[tile_plane + at] * w;
      sum.z += tile[2 * tile_plane + at] * w;
    }
    const float norm = 1.0F / total;
    store_rgb(p, x, y, make_float3(sum.x * norm, sum.y * norm, sum.z * norm));
  }
}

// À la taille de l'image : la conversion seule, le cas courant (registres de ce seul chemin).
__global__ void __launch_bounds__(kPixelThreads, kPixelBlocksPerSm) ingest(IngestParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.out_width || y >= p.out_height) return;
  store_rgb(p, x, y, rgb_at(p, x, y));
}

// À la taille de l'image, débandée dans la même passe (étape 5.4) : les cinq points de la croix du
// débandage (deband_kernels.cuh) sont convertis ici, arrondis en FP16 comme s'ils avaient été écrits par
// ingest puis relus par deband ; plus de tampon de travail ni de seconde passe. Noyau lourd (5 rgb_at).
__global__ void __launch_bounds__(kPixelThreads, kHeavyBlocksPerSm) ingest_deband(IngestParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.out_width || y >= p.out_height) return;
  const int2 o = deband_offset(x, y);
  const auto half_rgb = [&](int ox, int oy) {
    const float3 c = rgb_at(p, min(max(x + ox, 0), p.width - 1), min(max(y + oy, 0), p.height - 1));
    return make_float3(__half2float(__float2half_rn(c.x)), __half2float(__float2half_rn(c.y)),
                       __half2float(__float2half_rn(c.z)));
  };
  const float3 v = half_rgb(0, 0);
  const float3 a = half_rgb(o.x, o.y);
  const float3 b = half_rgb(-o.x, -o.y);
  const float3 d = half_rgb(-o.y, o.x);
  const float3 e = half_rgb(o.y, -o.x);
  store_rgb(p, x, y,
            make_float3(deband_rule(v.x, a.x, b.x, d.x, e.x), deband_rule(v.y, a.y, b.y, d.y, e.y),
                        deband_rule(v.z, a.z, b.z, d.z, e.z)));
}

// Réduction Area, en une passe (3 × 3 pixels de source au plus à 1,5). Noyau lourd : sm_120
// débordait de 8 octets sous 64 registres.
__global__ void __launch_bounds__(kPixelThreads, kHeavyBlocksPerSm) ingest_area(IngestParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.out_width || y >= p.out_height) return;
  store_rgb(p, x, y, area_at(p, x, y));
}

__global__ void __launch_bounds__(kPixelThreads, kPixelBlocksPerSm) write_rgba16(WriteParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) return;
  const std::size_t plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const std::size_t i =
      static_cast<std::size_t>(y) * static_cast<std::size_t>(p.width) + static_cast<std::size_t>(x);
  const auto* in = static_cast<const __half*>(p.rgb);
  store_ring(p.surface, x, y, in[i], in[plane + i], in[2 * plane + i]);
}

dim3 grid_for(int width, int height) {
  return dim3(static_cast<unsigned>((width + kBlock - 1) / kBlock),
              static_cast<unsigned>((height + kBlock - 1) / kBlock));
}

// Lignes de source qu'une tuile de kTileHeight lignes de sortie lit au plus : ses centres s'étendent sur
// (kTileHeight − 1) × échelle, plus le rayon étiré de part et d'autre, plus les arrondis (ceil, floor) et
// une marge pour les écarts d'un ulp entre l'hôte et l'appareil.
int tile_rows_for(const IngestParams& p) {
  const double scale = static_cast<double>(p.height) / static_cast<double>(p.out_height);
  const double radius =
      p.reduce_kind == kReduceTriangle ? 1.0 : (p.reduce_kind == kReduceSpline36 ? 3.0 : 2.0);
  return static_cast<int>((kTileHeight - 1) * scale + 2.0 * radius * scale) + 4;
}

// La passe tuilée tient-elle dans kMaxTileBytes de mémoire partagée ? Sinon, les deux passes.
constexpr std::size_t kMaxTileBytes = 40 * 1024;
std::size_t tile_bytes(const IngestParams& p) {
  return std::size_t{3} * static_cast<std::size_t>(tile_rows_for(p)) * kTileWidth * sizeof(float);
}

}  // namespace

std::size_t chroma_rows_bytes(const IngestParams& params) {
  if (params.chroma_layout != kChroma420) return 0;
  return std::size_t{2} * static_cast<std::size_t>(params.width) *
         static_cast<std::size_t>(params.chroma_height) * sizeof(float);
}

std::size_t ingest_rows_bytes(const IngestParams& params) {
  const bool same = params.out_width == params.width && params.out_height == params.height;
  if (same || params.reduce_kind == kReduceArea) return 0;
  if (tile_bytes(params) <= kMaxTileBytes) return 0;  // passe tuilée : pas de tampon (étape 5.3)
  return std::size_t{3} * static_cast<std::size_t>(params.out_width) *
         static_cast<std::size_t>(params.height) * sizeof(float);
}

cudaError_t launch_ingest(const IngestParams& params, cudaStream_t stream) {
  const dim3 block(kBlock, kBlock);
  const dim3 out = grid_for(params.out_width, params.out_height);
  if (params.chroma_layout != kChroma444 &&
      (params.x_start == nullptr || params.x_weights == nullptr || params.x_taps < 1 ||
       params.chroma_width < 1 || params.chroma_height < 1))
    return cudaErrorInvalidValue;
  if (params.chroma_layout == kChroma420) {
    if (params.y_start == nullptr || params.y_weights == nullptr || params.y_taps < 1 ||
        params.chroma_rows == nullptr)
      return cudaErrorInvalidValue;
    ingest_chroma_rows<<<grid_for(params.width, params.chroma_height), block, 0, stream>>>(params);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) return e;
  }
  if (params.out_width == params.width && params.out_height == params.height) {
    if (params.deband) {
      ingest_deband<<<out, block, 0, stream>>>(params);
      return cudaGetLastError();
    }
    ingest<<<out, block, 0, stream>>>(params);
    return cudaGetLastError();
  }
  if (params.reduce_kind == kReduceArea) {
    ingest_area<<<out, block, 0, stream>>>(params);
    return cudaGetLastError();
  }
  if (tile_bytes(params) <= kMaxTileBytes) {
    IngestParams tiled = params;
    tiled.tile_rows = tile_rows_for(params);
    const dim3 tiles(static_cast<unsigned>((params.out_width + kTileWidth - 1) / kTileWidth),
                     static_cast<unsigned>((params.out_height + kTileHeight - 1) / kTileHeight));
    ingest_tiled<<<tiles, dim3(kTileWidth, kTileThreadsY), tile_bytes(params), stream>>>(tiled);
    return cudaGetLastError();
  }
  // Au-delà (réduction de plus de ~4 fois en Spline36) : deux passes par le tampon rows.
  if (params.rows == nullptr) return cudaErrorInvalidValue;
  ingest_rows<<<grid_for(params.out_width, params.height), block, 0, stream>>>(params);
  if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) return e;
  ingest_columns<<<out, block, 0, stream>>>(params);
  return cudaGetLastError();
}

cudaError_t launch_write_rgba16(const WriteParams& params, cudaStream_t stream) {
  write_rgba16<<<grid_for(params.width, params.height), dim3(kBlock, kBlock), 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace ajp::gpu::cuda
