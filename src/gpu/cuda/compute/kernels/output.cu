// Sorties : RGB planaire vers YUV quantifié (P010, NV12, P016, planaires), éventuellement
// recadré (sortie de RIFE), et vers RGBA16 UNORM (surface ou mémoire linéaire).
// Repris de animejanai-inference-rtx@118c272 src/kernels.cu:221-313 (k_post_matrix,
// k_uv_v_store, k_post444) et 846-859 (k_rtx_pack : clamp, × crête, + 0,5, troncature),
// réécrits sur kernels/spec/color.h : Y quantifié aussitôt, Cb et Cr normalisés passent en
// horizontal puis en vertical, puis se quantifient avec l'arrondi pair exact de quantize().
// Tests : tests/unit/cuda_compute/output_test.cpp, rife_test.cpp.
#include "gpu/cuda/compute/kernels/device.cuh"
#include "gpu/cuda/compute/kernels/launch.h"

namespace ajp::gpu::cuda::dev {
namespace {

template <class T>
__device__ __forceinline__ float3 load_rgb(const Tensor& src, int x, int y) {
  const std::size_t plane = static_cast<std::size_t>(src.width) * static_cast<std::size_t>(src.height);
  const std::size_t i = at(x, y, src.width);
  const T* p = reinterpret_cast<const T*>(src.ptr);
  return make_float3(to_float(p[i]), to_float(p[plane + i]), to_float(p[2 * plane + i]));
}

// to_yuv() de spec/color.h, dans l'ordre écrit.
__device__ __forceinline__ float3 to_yuv(float3 p, const YuvToRgb& k) {
  const float y = k.kr * p.x + (1.0F - k.kr - k.kb) * p.y + k.kb * p.z;
  return make_float3(y, (p.z - y) / (2.0F * (1.0F - k.kb)), (p.x - y) / (2.0F * (1.0F - k.kr)));
}

__device__ __forceinline__ unsigned encode_luma(float y, const RgbToYuv& m) {
  return quantize(y * m.scale.yscale + m.scale.yoff, m.qdiv, m.qmax);
}

__device__ __forceinline__ unsigned encode_chroma(float c, const RgbToYuv& m) {
  return quantize(c * m.scale.cscale + m.scale.coff, m.qdiv, m.qmax);
}

template <class S>
__device__ __forceinline__ void store(Plane p, int x, int y, unsigned code) {
  mutable_row_of<S>(p, y)[x] = static_cast<S>(code);
}

// Matrice sur toute l'image : Y quantifié dans la fenêtre, Cb et Cr normalisés dans tmp.
template <class T, class S>
__global__ void k_matrix_420(Tensor src, RgbToYuv m, YuvTarget dst, float* tmp) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= src.width || y >= src.height) return;
  const float3 p = to_yuv(load_rgb<T>(src, x, y), m.scale);
  const std::size_t plane = static_cast<std::size_t>(src.width) * static_cast<std::size_t>(src.height);
  tmp[at(x, y, src.width)] = p.y;
  tmp[plane + at(x, y, src.width)] = p.z;
  const int wx = x - dst.left;
  const int wy = y - dst.top;
  if (wx >= 0 && wy >= 0 && wx < dst.width && wy < dst.height) store<S>(dst.y, wx, wy, encode_luma(p.x, m));
}

// Passe verticale de Cb et Cr (wy.src = hauteur de l'image), quantification et rangement, pour
// les seuls échantillons de la fenêtre (coin pair : chroma en left / 2, top / 2).
template <class S>
__global__ void k_chroma_store(const float* tmp, int width, Weights wy, RgbToYuv m, YuvTarget dst) {
  const int xw = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int yw = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (xw >= dst.width / 2 || yw >= dst.height / 2) return;
  const int x = dst.left / 2 + xw;
  const int y = dst.top / 2 + yw;
  const std::size_t plane = static_cast<std::size_t>(width) * static_cast<std::size_t>(wy.src);
  const float* w = wy.w + static_cast<std::size_t>(y) * static_cast<std::size_t>(wy.taps);
  const int s0 = wy.start[y];
  float u = 0.0F;
  float v = 0.0F;
  for (int j = 0; j < wy.taps; ++j) {
    const std::size_t i = at(x, mirror(s0 + j, wy.src), width);
    u += w[j] * tmp[i];
    v += w[j] * tmp[plane + i];
  }
  if (dst.interleaved) {
    S* row = mutable_row_of<S>(dst.u, yw);
    row[2 * xw] = static_cast<S>(encode_chroma(u, m));
    row[2 * xw + 1] = static_cast<S>(encode_chroma(v, m));
  } else {
    store<S>(dst.u, xw, yw, encode_chroma(u, m));
    store<S>(dst.v, xw, yw, encode_chroma(v, m));
  }
}

// 4:4:4 : matrice et quantification de la fenêtre, sans rééchantillonnage.
template <class T, class S>
__global__ void k_matrix_444(Tensor src, RgbToYuv m, YuvTarget dst) {
  const int xw = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int yw = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (xw >= dst.width || yw >= dst.height) return;
  const float3 p = to_yuv(load_rgb<T>(src, dst.left + xw, dst.top + yw), m.scale);
  store<S>(dst.y, xw, yw, encode_luma(p.x, m));
  store<S>(dst.u, xw, yw, encode_chroma(p.y, m));
  store<S>(dst.v, xw, yw, encode_chroma(p.z, m));
}

template <class T>
__device__ __forceinline__ ushort4 rgba16_at(const Tensor& src, int x, int y) {
  const float3 p = load_rgb<T>(src, x, y);
  return make_ushort4(unorm16(p.x), unorm16(p.y), unorm16(p.z), 65535);
}

template <class T>
__global__ void k_rgba16_surface(Tensor src, cudaSurfaceObject_t surface) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= src.width || y >= src.height) return;
  surf2Dwrite(rgba16_at<T>(src, x, y), surface, x * static_cast<int>(sizeof(ushort4)), y);
}

template <class T>
__global__ void k_rgba16_linear(Tensor src, Plane dst) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= src.width || y >= src.height) return;
  mutable_row_of<ushort4>(dst, y)[x] = rgba16_at<T>(src, x, y);
}

template <class T, class S>
void launch_matrix_420(Tensor src, RgbToYuv m, YuvTarget dst, float* tmp, cudaStream_t stream) {
  k_matrix_420<T, S><<<grid2d(src.width, src.height), block2d(), 0, stream>>>(src, m, dst, tmp);
}

template <class T, class S>
void launch_matrix_444(Tensor src, RgbToYuv m, YuvTarget dst, cudaStream_t stream) {
  k_matrix_444<T, S><<<grid2d(dst.width, dst.height), block2d(), 0, stream>>>(src, m, dst);
}

}  // namespace

cudaError_t output_matrix_420(Tensor src, RgbToYuv m, YuvTarget dst, float* tmp, cudaStream_t stream) {
  const bool u8 = dst.sample == Sample::U8;
  if (src.elem == Elem::F16) {
    u8 ? launch_matrix_420<__half, std::uint8_t>(src, m, dst, tmp, stream)
       : launch_matrix_420<__half, std::uint16_t>(src, m, dst, tmp, stream);
  } else {
    u8 ? launch_matrix_420<float, std::uint8_t>(src, m, dst, tmp, stream)
       : launch_matrix_420<float, std::uint16_t>(src, m, dst, tmp, stream);
  }
  return cudaGetLastError();
}

cudaError_t output_chroma_store(const float* tmp, int width, Weights wy, RgbToYuv m, YuvTarget dst,
                                cudaStream_t stream) {
  const dim3 grid = grid2d(dst.width / 2, dst.height / 2);
  if (dst.sample == Sample::U8) {
    k_chroma_store<std::uint8_t><<<grid, block2d(), 0, stream>>>(tmp, width, wy, m, dst);
  } else {
    k_chroma_store<std::uint16_t><<<grid, block2d(), 0, stream>>>(tmp, width, wy, m, dst);
  }
  return cudaGetLastError();
}

cudaError_t output_matrix_444(Tensor src, RgbToYuv m, YuvTarget dst, cudaStream_t stream) {
  const bool u8 = dst.sample == Sample::U8;
  if (src.elem == Elem::F16) {
    u8 ? launch_matrix_444<__half, std::uint8_t>(src, m, dst, stream)
       : launch_matrix_444<__half, std::uint16_t>(src, m, dst, stream);
  } else {
    u8 ? launch_matrix_444<float, std::uint8_t>(src, m, dst, stream)
       : launch_matrix_444<float, std::uint16_t>(src, m, dst, stream);
  }
  return cudaGetLastError();
}

cudaError_t rgba16_to_surface(Tensor src, cudaSurfaceObject_t surface, cudaStream_t stream) {
  const dim3 grid = grid2d(src.width, src.height);
  if (src.elem == Elem::F16) {
    k_rgba16_surface<__half><<<grid, block2d(), 0, stream>>>(src, surface);
  } else {
    k_rgba16_surface<float><<<grid, block2d(), 0, stream>>>(src, surface);
  }
  return cudaGetLastError();
}

cudaError_t rgba16_to_linear(Tensor src, Plane dst, cudaStream_t stream) {
  const dim3 grid = grid2d(src.width, src.height);
  if (src.elem == Elem::F16) {
    k_rgba16_linear<__half><<<grid, block2d(), 0, stream>>>(src, dst);
  } else {
    k_rgba16_linear<float><<<grid, block2d(), 0, stream>>>(src, dst);
  }
  return cudaGetLastError();
}

}  // namespace ajp::gpu::cuda::dev
