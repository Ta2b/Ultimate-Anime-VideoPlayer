// Entrée et sortie de RIFE : plans constants, instant t et recadrage de la fenêtre.
// Repris de animejanai-inference-rtx@118c272 src/kernels.cu:629-644 (k_rife_consts), réécrit sur
// kernels/spec/rife.h : grille 2i / (n − 1) − 1 et pas 2 / (n − 1), évalués en float dans
// l'ordre écrit. Tests : tests/unit/cuda_compute/rife_test.cpp.
#include "gpu/cuda/compute/kernels/device.cuh"
#include "gpu/cuda/compute/kernels/launch.h"

namespace ajp::gpu::cuda::dev {
namespace {

__device__ __forceinline__ float mesh(int i, int n) {
  return 2.0F * static_cast<float>(i) / static_cast<float>(n - 1) - 1.0F;
}

__device__ __forceinline__ float mesh_step(int n) {
  return 2.0F / static_cast<float>(n - 1);
}

template <class T>
__global__ void k_constants(Tensor first) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= first.width || y >= first.height) return;
  const std::size_t plane = static_cast<std::size_t>(first.width) * static_cast<std::size_t>(first.height);
  const std::size_t i = at(x, y, first.width);
  T* p = reinterpret_cast<T*>(first.ptr);
  p[i] = from_float<T>(mesh(x, first.width));
  p[plane + i] = from_float<T>(mesh(y, first.height));
  p[2 * plane + i] = from_float<T>(mesh_step(first.width));
  p[3 * plane + i] = from_float<T>(mesh_step(first.height));
}

template <class T>
__global__ void k_fill(T* p, std::size_t count, float value) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) p[i] = from_float<T>(value);
}

template <class S, class D>
__global__ void k_crop(Tensor src, int left, int top, Tensor dst) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= dst.width || y >= dst.height) return;
  const std::size_t z = blockIdx.z;
  const std::size_t src_plane = static_cast<std::size_t>(src.width) * static_cast<std::size_t>(src.height);
  const std::size_t dst_plane = static_cast<std::size_t>(dst.width) * static_cast<std::size_t>(dst.height);
  const S* in = reinterpret_cast<const S*>(src.ptr) + z * src_plane;
  D* out = reinterpret_cast<D*>(dst.ptr) + z * dst_plane;
  out[at(x, y, dst.width)] = from_float<D>(to_float(in[at(left + x, top + y, src.width)]));
}

template <class S>
void launch_crop(Tensor src, int left, int top, Tensor dst, cudaStream_t stream) {
  const dim3 grid = grid2d(dst.width, dst.height, 3);
  if (dst.elem == Elem::F16) {
    k_crop<S, __half><<<grid, block2d(), 0, stream>>>(src, left, top, dst);
  } else {
    k_crop<S, float><<<grid, block2d(), 0, stream>>>(src, left, top, dst);
  }
}

}  // namespace

cudaError_t rife_constants(Tensor first, cudaStream_t stream) {
  const dim3 grid = grid2d(first.width, first.height);
  if (first.elem == Elem::F16) {
    k_constants<__half><<<grid, block2d(), 0, stream>>>(first);
  } else {
    k_constants<float><<<grid, block2d(), 0, stream>>>(first);
  }
  return cudaGetLastError();
}

cudaError_t fill(std::uintptr_t ptr, Elem elem, std::size_t count, float value, cudaStream_t stream) {
  constexpr unsigned kThreads = 256;
  const auto blocks = static_cast<unsigned>((count + kThreads - 1) / kThreads);
  if (elem == Elem::F16) {
    k_fill<__half><<<blocks, kThreads, 0, stream>>>(reinterpret_cast<__half*>(ptr), count, value);
  } else {
    k_fill<float><<<blocks, kThreads, 0, stream>>>(reinterpret_cast<float*>(ptr), count, value);
  }
  return cudaGetLastError();
}

cudaError_t crop(Tensor src, int left, int top, Tensor dst, cudaStream_t stream) {
  if (src.elem == Elem::F16) {
    launch_crop<__half>(src, left, top, dst, stream);
  } else {
    launch_crop<float>(src, left, top, dst, stream);
  }
  return cudaGetLastError();
}

}  // namespace ajp::gpu::cuda::dev
