#pragma once
// Les deux noyaux de mise en page du SPAN entier en FP8, communs au réseau FP8 (span_fp8_net.cu) et au réseau
// de référence FP16 (span_fp8_ref.cu) : dégroupage 2×2 et DepthToSpace. Dans un espace de noms anonyme, pour
// que chaque unité de compilation garde les siens.
#include <cstddef>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <utility>

namespace ajp::gpu::cuda {
namespace {

// Mémoire d'appareil possédée.
class Device {
 public:
  Device() = default;
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  Device(Device&& other) noexcept : ptr_(std::exchange(other.ptr_, nullptr)) {}
  Device& operator=(Device&& other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = std::exchange(other.ptr_, nullptr);
    }
    return *this;
  }
  ~Device() { reset(); }
  void reset() {
    if (ptr_ != nullptr) cudaFree(ptr_);
    ptr_ = nullptr;
  }
  cudaError_t reserve(std::size_t bytes) {
    reset();
    const cudaError_t error = cudaMalloc(&ptr_, bytes);
    if (error != cudaSuccess) ptr_ = nullptr;
    return error;
  }
  cudaError_t upload(const void* host, std::size_t bytes) {
    const cudaError_t error = reserve(bytes);
    return error != cudaSuccess ? error : cudaMemcpy(ptr_, host, bytes, cudaMemcpyHostToDevice);
  }
  template <class T>
  const T* as() const {
    return static_cast<const T*>(ptr_);
  }
  void* get() const { return ptr_; }

 private:
  void* ptr_ = nullptr;
};

// Dégroupage : NCHW FP16 3×H×W -> NHWC 16 canaux à demi-résolution (canal = c·4 + a·2 + b ; 12 à 15 nuls).
__global__ void unshuffle_kernel(const __half* in, __half* out, int height, int width) {
  const int w = width / 2;
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= (height / 2) * w) return;
  const int y = idx / w, x = idx % w;
  __half v[16];
#pragma unroll
  for (int c = 0; c < 3; ++c)
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
      for (int b = 0; b < 2; ++b)
        v[c * 4 + a * 2 + b] =
            in[(static_cast<size_t>(c) * static_cast<size_t>(height) + static_cast<size_t>(2 * y + a)) *
                   static_cast<size_t>(width) +
               static_cast<size_t>(2 * x + b)];
#pragma unroll
  for (int i = 12; i < 16; ++i) v[i] = __float2half(0.F);
  uint4* o = reinterpret_cast<uint4*>(out + static_cast<size_t>(idx) * 16);
  o[0] = *reinterpret_cast<uint4*>(&v[0]);
  o[1] = *reinterpret_cast<uint4*>(&v[8]);
}

// DepthToSpace(4, mode CRD du modèle) : NHWC 48 canaux -> NCHW 3×4h×4w ; canal d'entrée = c·16 + bh·4 + bw.
__global__ void depth_to_space_kernel(const __half* in, __half* out, int h, int w) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);  // (pixel, bh, c)
  if (idx >= h * w * 12) return;
  const int c = idx % 3, bh = (idx / 3) % 4, pix = idx / 12;
  const int y = pix / w, x = pix % w;
  const __half* p = in + static_cast<size_t>(pix) * 48 + static_cast<size_t>(c * 16 + bh * 4);
  __half o[4];
#pragma unroll
  for (int bw = 0; bw < 4; ++bw) o[bw] = p[bw];
  *reinterpret_cast<uint2*>(
      out +
      (static_cast<size_t>(c) * static_cast<size_t>(4 * h) + static_cast<size_t>(4 * y + bh)) *
          static_cast<size_t>(4 * w) +
      static_cast<size_t>(4 * x)) = *reinterpret_cast<uint2*>(o);
}

}  // namespace
}  // namespace ajp::gpu::cuda
