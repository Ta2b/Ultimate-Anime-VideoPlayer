// Ingestion : YUV de l'appareil (NVDEC, pas de ligne quelconque) vers RGB planaire FP16 ou FP32.
// Repris de animejanai-inference-rtx@118c272 src/kernels.cu:101-216 (k_uv_h, k_v_f32,
// k_pre_combine, k_rife_uv_h, k_rife_pre_combine) et 334-386 (k_pre444, k_pre444_pad),
// réécrits sur kernels/spec/color.h et spec/resample.h : chroma brute en horizontal puis en
// vertical, puis normalisation et matrice. Le remplissage de RIFE est virtuel : hors de la
// source, les lectures rendent les codes du noir studio, comme rife_pad() de la référence.
// Tests : tests/unit/cuda_compute/ingest_test.cpp, rife_test.cpp.
#include "gpu/cuda/compute/kernels/device.cuh"
#include "gpu/cuda/compute/kernels/launch.h"

namespace ajp::gpu::cuda::dev {
namespace {

// Codes Cb et Cr du plan de chroma rempli en (cx, cy) : ceux de la source, ou pad_chroma.
template <class S>
__device__ __forceinline__ float2 chroma_at(const YuvSource& s, int cx, int cy) {
  const int d = s.subsampled ? 2 : 1;
  const int sx = cx - s.left / d;
  const int sy = cy - s.top / d;
  if (sx < 0 || sy < 0 || sx >= s.width / d || sy >= s.height / d) {
    const auto pad = static_cast<float>(s.pad_chroma);
    return make_float2(pad, pad);
  }
  if (s.interleaved) {
    const S* row = row_of<S>(s.u, sy);
    return make_float2(static_cast<float>(row[2 * sx]), static_cast<float>(row[2 * sx + 1]));
  }
  return make_float2(static_cast<float>(row_of<S>(s.u, sy)[sx]), static_cast<float>(row_of<S>(s.v, sy)[sx]));
}

template <class S>
__device__ __forceinline__ float luma_at(const YuvSource& s, int x, int y) {
  const int sx = x - s.left;
  const int sy = y - s.top;
  if (sx < 0 || sy < 0 || sx >= s.width || sy >= s.height) return static_cast<float>(s.pad_luma);
  return static_cast<float>(row_of<S>(s.y, sy)[sx]);
}

// normalize() puis to_rgb() de spec/color.h, dans l'ordre écrit.
__device__ __forceinline__ float3 to_rgb(float y, float u, float v, const YuvToRgb& m) {
  const float ny = (y - m.yoff) / m.yscale;
  const float nu = (u - m.coff) / m.cscale;
  const float nv = (v - m.coff) / m.cscale;
  const float kg = 1.0F - m.kr - m.kb;
  return make_float3(ny + 2.0F * (1.0F - m.kr) * nv,
                     ny - (2.0F * m.kb * (1.0F - m.kb) * nu + 2.0F * m.kr * (1.0F - m.kr) * nv) / kg,
                     ny + 2.0F * (1.0F - m.kb) * nu);
}

template <class T>
__device__ __forceinline__ void store_rgb(const Tensor& dst, int x, int y, float3 rgb) {
  const std::size_t plane = static_cast<std::size_t>(dst.width) * static_cast<std::size_t>(dst.height);
  const std::size_t i = at(x, y, dst.width);
  T* p = reinterpret_cast<T*>(dst.ptr);
  p[i] = from_float<T>(rgb.x);
  p[plane + i] = from_float<T>(rgb.y);
  p[2 * plane + i] = from_float<T>(rgb.z);
}

// Passe horizontale de la chroma 4:2:0 rangée en codes bruts : wx.src = padded_width / 2.
template <class S>
__global__ void k_chroma_rows(YuvSource s, Weights wx, float* tmp) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int rows = s.padded_height / 2;
  if (x >= wx.dst || y >= rows) return;
  const float* w = wx.w + static_cast<std::size_t>(x) * static_cast<std::size_t>(wx.taps);
  const int s0 = wx.start[x];
  float u = 0.0F;
  float v = 0.0F;
  for (int j = 0; j < wx.taps; ++j) {
    const float2 c = chroma_at<S>(s, mirror(s0 + j, wx.src), y);
    u += w[j] * c.x;
    v += w[j] * c.y;
  }
  const std::size_t plane = static_cast<std::size_t>(wx.dst) * static_cast<std::size_t>(rows);
  tmp[at(x, y, wx.dst)] = u;
  tmp[plane + at(x, y, wx.dst)] = v;
}

// Passe verticale de la chroma (wy.src = padded_height / 2), puis matrice avec la luma.
template <class S, class T>
__global__ void k_combine_420(YuvSource s, const float* tmp, Weights wy, YuvToRgb m, Tensor dst) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= dst.width || y >= dst.height) return;
  const std::size_t plane = static_cast<std::size_t>(dst.width) * static_cast<std::size_t>(wy.src);
  const float* w = wy.w + static_cast<std::size_t>(y) * static_cast<std::size_t>(wy.taps);
  const int s0 = wy.start[y];
  float u = 0.0F;
  float v = 0.0F;
  for (int j = 0; j < wy.taps; ++j) {
    const std::size_t i = at(x, mirror(s0 + j, wy.src), dst.width);
    u += w[j] * tmp[i];
    v += w[j] * tmp[plane + i];
  }
  store_rgb<T>(dst, x, y, to_rgb(luma_at<S>(s, x, y), u, v, m));
}

template <class S, class T>
__global__ void k_combine_444(YuvSource s, YuvToRgb m, Tensor dst) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= dst.width || y >= dst.height) return;
  const float2 c = chroma_at<S>(s, x, y);
  store_rgb<T>(dst, x, y, to_rgb(luma_at<S>(s, x, y), c.x, c.y, m));
}

__global__ void k_probe() {}

}  // namespace

cudaError_t probe_kernel_image() {
  cudaFuncAttributes attributes{};
  return cudaFuncGetAttributes(&attributes, k_probe);
}

cudaError_t ingest_chroma_rows(const YuvSource& src, Weights wx, float* tmp, cudaStream_t stream) {
  const dim3 grid = grid2d(wx.dst, src.padded_height / 2);
  if (src.sample == Sample::U8) {
    k_chroma_rows<std::uint8_t><<<grid, block2d(), 0, stream>>>(src, wx, tmp);
  } else {
    k_chroma_rows<std::uint16_t><<<grid, block2d(), 0, stream>>>(src, wx, tmp);
  }
  return cudaGetLastError();
}

cudaError_t ingest_combine_420(const YuvSource& src, const float* tmp, Weights wy, YuvToRgb m, Tensor dst,
                               cudaStream_t stream) {
  const dim3 grid = grid2d(dst.width, dst.height);
  const bool u8 = src.sample == Sample::U8;
  if (dst.elem == Elem::F16) {
    if (u8) {
      k_combine_420<std::uint8_t, __half><<<grid, block2d(), 0, stream>>>(src, tmp, wy, m, dst);
    } else {
      k_combine_420<std::uint16_t, __half><<<grid, block2d(), 0, stream>>>(src, tmp, wy, m, dst);
    }
  } else if (u8) {
    k_combine_420<std::uint8_t, float><<<grid, block2d(), 0, stream>>>(src, tmp, wy, m, dst);
  } else {
    k_combine_420<std::uint16_t, float><<<grid, block2d(), 0, stream>>>(src, tmp, wy, m, dst);
  }
  return cudaGetLastError();
}

cudaError_t ingest_combine_444(const YuvSource& src, YuvToRgb m, Tensor dst, cudaStream_t stream) {
  const dim3 grid = grid2d(dst.width, dst.height);
  const bool u8 = src.sample == Sample::U8;
  if (dst.elem == Elem::F16) {
    if (u8) {
      k_combine_444<std::uint8_t, __half><<<grid, block2d(), 0, stream>>>(src, m, dst);
    } else {
      k_combine_444<std::uint16_t, __half><<<grid, block2d(), 0, stream>>>(src, m, dst);
    }
  } else if (u8) {
    k_combine_444<std::uint8_t, float><<<grid, block2d(), 0, stream>>>(src, m, dst);
  } else {
    k_combine_444<std::uint16_t, float><<<grid, block2d(), 0, stream>>>(src, m, dst);
  }
  return cudaGetLastError();
}

}  // namespace ajp::gpu::cuda::dev
