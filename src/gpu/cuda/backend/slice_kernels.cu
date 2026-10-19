// Noyaux provisoires de la tranche verticale (voir slice_kernels.h).
#include <cuda_fp16.h>

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
  const float top = sample(p.chroma, p.chroma_pitch, 2 * x0 + c, y0, p.sixteen_bits) * (1.0F - ax) +
                    sample(p.chroma, p.chroma_pitch, 2 * x1 + c, y0, p.sixteen_bits) * ax;
  const float bottom = sample(p.chroma, p.chroma_pitch, 2 * x0 + c, y1, p.sixteen_bits) * (1.0F - ax) +
                       sample(p.chroma, p.chroma_pitch, 2 * x1 + c, y1, p.sixteen_bits) * ax;
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
  for (int j = static_cast<int>(floorf(y0)); j < last_y; ++j) {
    const float wy = fminf(y1, static_cast<float>(j + 1)) - fmaxf(y0, static_cast<float>(j));
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

// Pixel de sortie (x, y) par un filtre séparable étiré à l'échelle de la réduction (anticrénelé,
// comme la réduction de mpv) : chaque pixel de source à moins du rayon × échelle du centre, poids
// normalisés ; bords répétés.
__device__ float3 filtered_at(const IngestParams& p, int x, int y) {
  const float sx = static_cast<float>(p.width) / static_cast<float>(p.out_width);
  const float sy = static_cast<float>(p.height) / static_cast<float>(p.out_height);
  const float radius = p.reduce_kind == kReduceTriangle   ? 1.0F
                       : p.reduce_kind == kReduceSpline36 ? 3.0F
                                                          : 2.0F;
  const float cx = (static_cast<float>(x) + 0.5F) * sx - 0.5F;
  const float cy = (static_cast<float>(y) + 0.5F) * sy - 0.5F;
  const int i0 = static_cast<int>(ceilf(cx - radius * sx));
  const int j0 = static_cast<int>(ceilf(cy - radius * sy));
  const int nx = min(static_cast<int>(floorf(cx + radius * sx)) - i0 + 1, kMaxReduceTaps);
  const int ny = min(static_cast<int>(floorf(cy + radius * sy)) - j0 + 1, kMaxReduceTaps);
  float wx[kMaxReduceTaps];
  float total_x = 0.0F;
  for (int i = 0; i < nx; ++i) {
    wx[i] = reduce_weight(p, (static_cast<float>(i0 + i) - cx) / sx);
    total_x += wx[i];
  }
  float3 sum = make_float3(0.0F, 0.0F, 0.0F);
  float total_y = 0.0F;
  for (int j = 0; j < ny; ++j) {
    const float wy = reduce_weight(p, (static_cast<float>(j0 + j) - cy) / sy);
    if (wy == 0.0F) continue;
    total_y += wy;
    const int row = clamp_index(j0 + j, p.height);
    float3 line = make_float3(0.0F, 0.0F, 0.0F);
    for (int i = 0; i < nx; ++i) {
      if (wx[i] == 0.0F) continue;
      const float3 c = rgb_at(p, clamp_index(i0 + i, p.width), row);
      line.x += c.x * wx[i];
      line.y += c.y * wx[i];
      line.z += c.z * wx[i];
    }
    sum.x += line.x * wy;
    sum.y += line.y * wy;
    sum.z += line.z * wy;
  }
  const float norm = 1.0F / (total_x * total_y);
  return make_float3(sum.x * norm, sum.y * norm, sum.z * norm);
}

__global__ void ingest(IngestParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.out_width || y >= p.out_height) return;
  const bool same = p.out_width == p.width && p.out_height == p.height;
  float3 c;
  if (same) {
    c = rgb_at(p, x, y);
  } else if (p.reduce_kind == kReduceArea) {
    c = area_at(p, x, y);
  } else {
    c = filtered_at(p, x, y);
  }
  const std::size_t plane = static_cast<std::size_t>(p.out_width) * static_cast<std::size_t>(p.out_height);
  const std::size_t i =
      static_cast<std::size_t>(y) * static_cast<std::size_t>(p.out_width) + static_cast<std::size_t>(x);
  auto* out = static_cast<__half*>(p.rgb);
  out[i] = __float2half_rn(c.x);
  out[plane + i] = __float2half_rn(c.y);
  out[2 * plane + i] = __float2half_rn(c.z);
}

// kernels/spec/color.h unorm16 : clamp, × 65535, + 0,5, troncature.
__device__ unsigned short unorm16(float v) {
  v = v != v ? 0.0F : fminf(fmaxf(v, 0.0F), 1.0F);  // NaN donne 0
  return static_cast<unsigned short>(v * 65535.0F + 0.5F);
}

__global__ void write_rgba16(WriteParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) return;
  const std::size_t plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const std::size_t i =
      static_cast<std::size_t>(y) * static_cast<std::size_t>(p.width) + static_cast<std::size_t>(x);
  const auto* in = static_cast<const __half*>(p.rgb);
  const ushort4 pixel = make_ushort4(unorm16(__half2float(in[i])), unorm16(__half2float(in[plane + i])),
                                     unorm16(__half2float(in[2 * plane + i])), 65535);
  surf2Dwrite(pixel, p.surface, x * static_cast<int>(sizeof(ushort4)), y);
}

dim3 grid_for(int width, int height) {
  return dim3(static_cast<unsigned>((width + kBlock - 1) / kBlock),
              static_cast<unsigned>((height + kBlock - 1) / kBlock));
}

}  // namespace

cudaError_t launch_ingest(const IngestParams& params, cudaStream_t stream) {
  ingest<<<grid_for(params.out_width, params.out_height), dim3(kBlock, kBlock), 0, stream>>>(params);
  return cudaGetLastError();
}

cudaError_t launch_write_rgba16(const WriteParams& params, cudaStream_t stream) {
  write_rgba16<<<grid_for(params.width, params.height), dim3(kBlock, kBlock), 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace ajp::gpu::cuda
