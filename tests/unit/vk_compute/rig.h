// Banc des tests de vk_compute : un appareil créé (lavapipe en CI), l'allocateur, le cache de
// pipelines et le KernelRunner, plus l'envoi et la lecture des images de kernels/image.h.
// Sans Vulkan (bibliothèque ou appareil absents), le test est sauté avec la raison ; tout autre
// refus le fait échouer.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <format>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/types.h"
#include "gpu/vulkan/compute/buffer.h"
#include "gpu/vulkan/compute/context.h"
#include "gpu/vulkan/compute/kernel_runner.h"
#include "gpu/vulkan/compute/kernel_types.h"
#include "gpu/vulkan/compute/memory.h"
#include "gpu/vulkan/compute/pipeline.h"
#include "gpu/vulkan/compute/recording.h"
#include "kernels/image.h"

namespace ajp::gpu::vk::test {

std::string describe(const Refusal& refusal);

// Le succès, ou l'échec du test avec le refus.
inline void require_ok(const Done& done) {
  if (!done) FAIL(describe(done.error()));
}

template <class T>
T require_value(std::expected<T, Refusal> result) {
  if (!result) FAIL(describe(result.error()));
  return std::move(*result);
}

// Le refus attendu, avec son code.
template <class T>
void require_refusal(const std::expected<T, Refusal>& result, RefusalCode code) {
  REQUIRE_FALSE(result.has_value());
  INFO(describe(result.error()));
  CHECK(result.error().code == code);
}

struct Rig {
  std::unique_ptr<VkContext> context;
  std::unique_ptr<MemoryAllocator> memory;
  std::unique_ptr<PipelineCache> pipelines;
  std::optional<KernelRunner> runner;

  KernelRunner& kernels() { return *runner; }
};

// Saute le test sans Vulkan ; tout autre refus le fait échouer.
Rig make_rig(const CreateOptions& options = {});
// Saute le test sans stockage 16 bits.
void require_f16(const Rig& rig);

// Enregistre, soumet une fois, attend sur l'hôte ; l'enregistrement reste rejouable.
std::unique_ptr<Recording> record(Rig& rig, const std::function<Done(Recorder&)>& body);
void run(Rig& rig, Recording& recording);
void record_and_run(Rig& rig, const std::function<Done(Recorder&)>& body);

Buffer make_buffer(Rig& rig, VkDeviceSize bytes, MemoryUse use = MemoryUse::Device);
void write_bytes(Rig& rig, const Buffer& buffer, std::span<const std::byte> bytes, VkDeviceSize offset = 0);
std::vector<std::byte> read_bytes(Rig& rig, const Buffer& buffer, std::size_t count, VkDeviceSize offset = 0);

template <class T>
std::vector<T> read_as(Rig& rig, const Buffer& buffer, std::size_t count) {
  const std::vector<std::byte> bytes = read_bytes(rig, buffer, count * sizeof(T));
  std::vector<T> out(count);
  std::memcpy(out.data(), bytes.data(), bytes.size());
  return out;
}

// Une image YUV dans son tampon, en disposition serrée.
struct YuvBuffer {
  Buffer buffer;
  YuvView view;
};

std::vector<std::byte> pack(const kernels::YuvImage& image, const YuvLayout& layout);
kernels::YuvImage unpack(std::span<const std::byte> bytes, const YuvView& view);
YuvBuffer upload_yuv(Rig& rig, const kernels::YuvImage& image);
// Cible remplie d'octets 0xA5 : un plan que le noyau n'écrit pas se voit.
YuvBuffer yuv_target(Rig& rig, kernels::Format format, std::uint32_t width, std::uint32_t height);
kernels::YuvImage download_yuv(Rig& rig, const YuvBuffer& image);

// Un tenseur planaire dans son tampon.
struct TensorBuffer {
  Buffer buffer;
  TensorView view;
};

TensorBuffer upload_planes(Rig& rig, std::span<const kernels::PlaneF> planes, Precision precision);
TensorBuffer upload_rgb(Rig& rig, const kernels::RgbImage& image, Precision precision);
TensorBuffer tensor_target(Rig& rig, std::uint32_t width, std::uint32_t height, std::uint32_t planes,
                           Precision precision);
std::vector<kernels::PlaneF> download_planes(Rig& rig, const TensorBuffer& tensor);
kernels::RgbImage download_rgb(Rig& rig, const TensorBuffer& tensor);

// Images synthétiques déterministes : bruit, rampes et codes extrêmes du format.
kernels::YuvImage synthetic_yuv(kernels::Format format, int width, int height, std::uint32_t seed);
kernels::RgbImage synthetic_rgb(int width, int height, std::uint32_t seed, float lo = -0.1F, float hi = 1.1F);
kernels::PlaneF synthetic_plane(int width, int height, std::uint32_t seed, float lo, float hi);

// Ce qu'un port relit d'une image rangée en float16 : la référence reçoit la même entrée.
kernels::RgbImage as_stored(const kernels::RgbImage& image, Precision precision);

float max_abs_diff(const kernels::PlaneF& a, const kernels::PlaneF& b);
float max_abs_diff(const kernels::RgbImage& a, const kernels::RgbImage& b);
int max_code_diff(const kernels::PlaneU16& a, const kernels::PlaneU16& b);

}  // namespace ajp::gpu::vk::test
