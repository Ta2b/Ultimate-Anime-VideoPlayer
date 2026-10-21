// Noyaux provisoires de la tranche verticale (voir slice_kernels.h).
#include <cuda_fp16.h>

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

// Chroma bilinéaire au point (cx, cy) du plan de chroma, bords répétés ; c vaut 0 (Cb) ou 1 (Cr).
__device__ float chroma_at(const IngestParams& p, float cx, float cy, int c) {
  const int cw = (p.width + 1) / 2;
  const int ch = (p.height + 1) / 2;
  const float fx = floorf(cx);
  const float fy = floorf(cy);
  const float ax = cx - fx;
  const float ay = cy - fy;
  const int x0 = clamp_index(static_cast<int>(fx), cw);
  const int x1 = clamp_index(static_cast<int>(fx) + 1, cw);
  const int y0 = clamp_index(static_cast<int>(fy), ch);
  const int y1 = clamp_index(static_cast<int>(fy) + 1, ch);
  // Entrelacée : Cb et Cr alternent dans chroma. Planaire : Cb dans chroma, Cr dans chroma_v.
  const bool planar = p.chroma_v != nullptr;
  const void* plane = planar && c == 1 ? p.chroma_v : p.chroma;
  const int stride = planar ? 1 : 2;
  const int lane = planar ? 0 : c;
  const float top = sample(plane, p.chroma_pitch, stride * x0 + lane, y0, p.sixteen_bits) * (1.0F - ax) +
                    sample(plane, p.chroma_pitch, stride * x1 + lane, y0, p.sixteen_bits) * ax;
  const float bottom = sample(plane, p.chroma_pitch, stride * x0 + lane, y1, p.sixteen_bits) * (1.0F - ax) +
                       sample(plane, p.chroma_pitch, stride * x1 + lane, y1, p.sixteen_bits) * ax;
  return top * (1.0F - ay) + bottom * ay;
}

// RGB du pixel (x, y) de l'image décodée.
__device__ float3 rgb_at(const IngestParams& p, int x, int y) {
  const float cx = (static_cast<float>(x) - p.chroma_x) * 0.5F;
  const float cy = (static_cast<float>(y) - p.chroma_y) * 0.5F;
  const float luma = (sample(p.luma, p.luma_pitch, x, y, p.sixteen_bits) - p.yoff) / p.yscale;
  const float cb = (chroma_at(p, cx, cy, 0) - p.coff) / p.cscale;
  const float cr = (chroma_at(p, cx, cy, 1) - p.coff) / p.cscale;
  const float kg = 1.0F - p.kr - p.kb;
  return make_float3(luma + 2.0F * (1.0F - p.kr) * cr,
                     luma - (2.0F * p.kb * (1.0F - p.kb) * cb + 2.0F * p.kr * (1.0F - p.kr) * cr) / kg,
                     luma + 2.0F * (1.0F - p.kb) * cb);
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
  const int first = static_cast<int>(ceilf(center - radius * scale));
  const int count = min(static_cast<int>(floorf(center + radius * scale)) - first + 1, kMaxReduceTaps);
  return Taps{.first = first, .count = count, .center = center, .scale = scale};
}

__device__ float tap_weight(const IngestParams& p, const Taps& t, int i) {
  return reduce_weight(p, (static_cast<float>(t.first + i) - t.center) / t.scale);
}

// Première passe des filtres étirés : la ligne y de la source, convertie en RGB et filtrée vers
// out_width colonnes, poids normalisés, bords répétés. Boucles non déroulées : sm_120 déroule
// sinon les 32 prises et déborde des 64 registres.
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

// À la taille de l'image : la conversion seule, le cas courant (registres de ce seul chemin).
__global__ void __launch_bounds__(kPixelThreads, kPixelBlocksPerSm) ingest(IngestParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.out_width || y >= p.out_height) return;
  store_rgb(p, x, y, rgb_at(p, x, y));
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
  store_rgba16(p.surface, x, y, in[i], in[plane + i], in[2 * plane + i]);
}

dim3 grid_for(int width, int height) {
  return dim3(static_cast<unsigned>((width + kBlock - 1) / kBlock),
              static_cast<unsigned>((height + kBlock - 1) / kBlock));
}

}  // namespace

std::size_t ingest_rows_bytes(const IngestParams& params) {
  const bool same = params.out_width == params.width && params.out_height == params.height;
  if (same || params.reduce_kind == kReduceArea) return 0;
  return std::size_t{3} * static_cast<std::size_t>(params.out_width) *
         static_cast<std::size_t>(params.height) * sizeof(float);
}

cudaError_t launch_ingest(const IngestParams& params, cudaStream_t stream) {
  const dim3 block(kBlock, kBlock);
  const dim3 out = grid_for(params.out_width, params.out_height);
  if (params.out_width == params.width && params.out_height == params.height) {
    ingest<<<out, block, 0, stream>>>(params);
    return cudaGetLastError();
  }
  if (params.reduce_kind == kReduceArea) {
    ingest_area<<<out, block, 0, stream>>>(params);
    return cudaGetLastError();
  }
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
