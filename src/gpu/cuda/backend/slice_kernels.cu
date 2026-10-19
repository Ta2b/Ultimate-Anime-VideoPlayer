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

__global__ void ingest(IngestParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) return;
  const float cx = (static_cast<float>(x) - p.chroma_x) * 0.5F;
  const float cy = (static_cast<float>(y) - p.chroma_y) * 0.5F;
  const float luma = (sample(p.luma, p.luma_pitch, x, y, p.sixteen_bits) - p.yoff) / p.yscale;
  const float cb = (chroma_at(p, cx, cy, 0) - p.coff) / p.cscale;
  const float cr = (chroma_at(p, cx, cy, 1) - p.coff) / p.cscale;
  const float kg = 1.0F - p.kr - p.kb;
  const float r = luma + 2.0F * (1.0F - p.kr) * cr;
  const float g = luma - (2.0F * p.kb * (1.0F - p.kb) * cb + 2.0F * p.kr * (1.0F - p.kr) * cr) / kg;
  const float b = luma + 2.0F * (1.0F - p.kb) * cb;
  const std::size_t plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const std::size_t i =
      static_cast<std::size_t>(y) * static_cast<std::size_t>(p.width) + static_cast<std::size_t>(x);
  auto* out = static_cast<__half*>(p.rgb);
  out[i] = __float2half_rn(r);
  out[plane + i] = __float2half_rn(g);
  out[2 * plane + i] = __float2half_rn(b);
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
  ingest<<<grid_for(params.width, params.height), dim3(kBlock, kBlock), 0, stream>>>(params);
  return cudaGetLastError();
}

cudaError_t launch_write_rgba16(const WriteParams& params, cudaStream_t stream) {
  write_rgba16<<<grid_for(params.width, params.height), dim3(kBlock, kBlock), 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace ajp::gpu::cuda
