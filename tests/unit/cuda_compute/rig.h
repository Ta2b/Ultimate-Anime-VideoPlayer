// Banc des tests CUDA : le premier appareil CUDA, un pool et un flux, et les transferts entre
// les images de la référence (kernels/image.h) et la mémoire de l'appareil. Sans GPU NVIDIA, les
// tests se SKIP avec la raison (la CI n'en a pas).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "gpu/cuda/compute/device.h"
#include "gpu/cuda/compute/image.h"
#include "gpu/cuda/compute/memory.h"
#include "kernels/image.h"

namespace ajp::gpu::cuda::test {

struct Rig {
  Device device;
  std::unique_ptr<DevicePool> pool;
  Stream stream;

  cudaStream_t s() const { return stream.get(); }

  // Une copie de host dans un tampon du pool (synchrone).
  DeviceBuffer upload(const void* host, std::size_t bytes);
  template <class T>
  DeviceBuffer upload(const std::vector<T>& host) {
    return upload(host.data(), host.size() * sizeof(T));
  }
  // bytes octets relus après la fin du flux.
  std::vector<std::byte> download(const void* source, std::size_t bytes);
  template <class T>
  std::vector<T> download(const DeviceBuffer& buffer, std::size_t count) {
    const std::vector<std::byte> bytes = download(buffer.data(), count * sizeof(T));
    std::vector<T> out(count);
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
  }
};

// Le banc sur le premier appareil CUDA, ou la raison de son absence.
struct RigOrReason {
  std::optional<Rig> rig;
  std::string reason;
};
RigOrReason open_rig(std::size_t pool_bytes = std::size_t{512} << 20);

// Une image YUV de la référence rangée sur l'appareil, avec un pas de ligne plus long que la
// ligne (comme NVDEC) et la chroma entrelacée si le format l'est.
struct DeviceYuv {
  std::vector<DeviceBuffer> buffers;
  YuvPlanes planes;
};
DeviceYuv upload_yuv(Rig& rig, const kernels::YuvImage& image);
// Une image YUV vide du format et de la taille donnés.
DeviceYuv alloc_yuv(Rig& rig, kernels::Format format, int width, int height);
kernels::YuvImage download_yuv(Rig& rig, const DeviceYuv& image);

// Un tenseur planaire sur l'appareil, et sa mémoire.
struct DeviceTensor {
  DeviceBuffer buffer;
  Tensor tensor;
};
DeviceTensor alloc_tensor(Rig& rig, Elem elem, int width, int height, int channels = 3);
// Plans de même taille, rangés en float ou en fp16 (au plus proche, égalités au pair).
DeviceTensor upload_planes(Rig& rig, const std::vector<kernels::PlaneF>& planes, Elem elem);
DeviceTensor upload_rgb(Rig& rig, const kernels::RgbImage& image, Elem elem);
std::vector<kernels::PlaneF> download_planes(Rig& rig, const DeviceTensor& t);
kernels::RgbImage download_rgb(Rig& rig, const DeviceTensor& t);

// Images aléatoires reproductibles : codes valides du format, RGB dans [lo, hi].
kernels::YuvImage random_yuv(kernels::Format format, int width, int height, std::uint32_t seed);
kernels::RgbImage random_rgb(int width, int height, float lo, float hi, std::uint32_t seed);
kernels::PlaneF random_plane(int width, int height, float lo, float hi, std::uint32_t seed);

// Plus grand écart absolu entre deux plans de même taille ; +inf si un seul est NaN.
float max_diff(const kernels::PlaneF& a, const kernels::PlaneF& b);
float max_diff(const kernels::RgbImage& a, const kernels::RgbImage& b);

}  // namespace ajp::gpu::cuda::test

// SKIP avec la raison quand il n'y a pas d'appareil CUDA ; sinon rig est le banc ouvert.
#define AJP_CUDA_RIG(rig, ...)                                     \
  auto rig##_open = ::ajp::gpu::cuda::test::open_rig(__VA_ARGS__); \
  if (!rig##_open.rig) SKIP(rig##_open.reason);                    \
  ::ajp::gpu::cuda::test::Rig& rig = *rig##_open.rig
