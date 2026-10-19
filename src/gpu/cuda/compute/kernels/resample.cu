// Passes séparables de rééchantillonnage sur des tenseurs planaires : horizontale vers un
// intermédiaire float, puis verticale vers le tenseur de sortie (FP16 ou FP32).
// Repris de animejanai-inference-rtx@118c272 src/kernels.cu:244-259 (k_h_f32) et 415-445
// (k_rs_h, k_rs_v), réécrits sur kernels/spec/resample.h : somme float de j = 0 à taps − 1,
// depuis 0, indices source passés par le miroir, rien n'est borné entre les passes.
// Tests : tests/unit/cuda_compute/resize_test.cpp.
#include "gpu/cuda/compute/kernels/device.cuh"
#include "gpu/cuda/compute/kernels/launch.h"

namespace ajp::gpu::cuda::dev {
namespace {

template <class T>
__global__ void k_rows(Tensor src, Weights wx, float* tmp) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= wx.dst || y >= src.height) return;
  const std::size_t z = blockIdx.z;
  const T* row = reinterpret_cast<const T*>(src.ptr) +
                 z * static_cast<std::size_t>(src.width) * static_cast<std::size_t>(src.height) +
                 static_cast<std::size_t>(y) * static_cast<std::size_t>(src.width);
  const float* w = wx.w + static_cast<std::size_t>(x) * static_cast<std::size_t>(wx.taps);
  const int s0 = wx.start[x];
  float a = 0.0F;
  for (int j = 0; j < wx.taps; ++j) a += w[j] * to_float(row[mirror(s0 + j, wx.src)]);
  tmp[z * static_cast<std::size_t>(wx.dst) * static_cast<std::size_t>(src.height) + at(x, y, wx.dst)] = a;
}

template <class T>
__global__ void k_columns(const float* tmp, int width, Weights wy, Tensor dst) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= width || y >= wy.dst) return;
  const std::size_t z = blockIdx.z;
  const float* plane = tmp + z * static_cast<std::size_t>(width) * static_cast<std::size_t>(wy.src);
  const float* w = wy.w + static_cast<std::size_t>(y) * static_cast<std::size_t>(wy.taps);
  const int s0 = wy.start[y];
  float a = 0.0F;
  for (int j = 0; j < wy.taps; ++j) a += w[j] * plane[at(x, mirror(s0 + j, wy.src), width)];
  T* out =
      reinterpret_cast<T*>(dst.ptr) + z * static_cast<std::size_t>(width) * static_cast<std::size_t>(wy.dst);
  out[at(x, y, width)] = from_float<T>(a);
}

}  // namespace

cudaError_t rows(Tensor src, int planes, Weights wx, float* tmp, cudaStream_t stream) {
  const dim3 grid = grid2d(wx.dst, src.height, planes);
  if (src.elem == Elem::F16) {
    k_rows<__half><<<grid, block2d(), 0, stream>>>(src, wx, tmp);
  } else {
    k_rows<float><<<grid, block2d(), 0, stream>>>(src, wx, tmp);
  }
  return cudaGetLastError();
}

cudaError_t columns(const float* tmp, int width, int planes, Weights wy, Tensor dst, cudaStream_t stream) {
  const dim3 grid = grid2d(width, wy.dst, planes);
  if (dst.elem == Elem::F16) {
    k_columns<__half><<<grid, block2d(), 0, stream>>>(tmp, width, wy, dst);
  } else {
    k_columns<float><<<grid, block2d(), 0, stream>>>(tmp, width, wy, dst);
  }
  return cudaGetLastError();
}

}  // namespace ajp::gpu::cuda::dev
