#pragma once
// Le noyau de mise en page du SPAN entier en FP8, commun au réseau FP8 (span_fp8_net.cu) et au réseau de
// référence FP16 (span_fp8_ref.cu) : dégroupage 2×2. DepthToSpace est fondu à l'épilogue de to_pixel dans le
// premier (étape 3.4a) et reste un noyau à part dans la seconde. Dans un espace de noms anonyme, pour
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

}  // namespace
}  // namespace ajp::gpu::cuda
