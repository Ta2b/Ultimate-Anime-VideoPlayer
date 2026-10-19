// Banc des tests du backend Vulkan : un VulkanBackend sur l'appareil du système (sauté sans
// Vulkan), l'envoi et la lecture de ses tenseurs, des images décodées en mémoire hôte, et des
// modèles ONNX écrits dans un dossier temporaire.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/frame.h"
#include "gpu/backend.h"
#include "gpu/vulkan/backend/vulkan_backend.h"
#include "kernels/image.h"

namespace ajp::gpu::vk::test {

// Un dossier temporaire neuf, supprimé à la destruction.
class TempDir {
 public:
  explicit TempDir(std::string_view name);
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  TempDir(TempDir&&) = delete;
  TempDir& operator=(TempDir&&) = delete;
  ~TempDir();
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

// Saute sans Vulkan ; tout autre refus fait échouer le test.
std::unique_ptr<VulkanBackend> make_backend(const std::filesystem::path& cache_dir, BackendOptions options = {});

// Octets d'un tenseur du backend, écrits et lus par une copie attendue sur l'hôte, après la fin
// de tout ce que la file du backend a reçu.
void write_tensor(VulkanBackend& backend, const Tensor& tensor, std::span<const std::byte> bytes);
std::vector<std::byte> read_tensor(VulkanBackend& backend, const Tensor& tensor, std::size_t bytes);

// Image RGB planaire en tenseur 1 × 3 × h × w, float ou float16 (arrondi au plus proche).
std::vector<std::byte> rgb_bytes(const kernels::RgbImage& image, TensorType type);
kernels::RgbImage rgb_from(std::span<const std::byte> bytes, int width, int height, TensorType type);
std::vector<std::byte> planes_bytes(std::span<const kernels::PlaneF> planes, TensorType type);
std::unique_ptr<TensorBuffer> rgb_tensor(VulkanBackend& backend, int width, int height, TensorType type,
                                         int planes = 3);
// Ce qu'un tenseur float16 garde d'une image.
kernels::RgbImage stored(const kernels::RgbImage& image, TensorType type);

// Image YUV synthétique : bruit, rampes et codes extrêmes du format.
kernels::YuvImage synthetic_yuv(kernels::Format format, int width, int height, std::uint32_t seed);
kernels::RgbImage synthetic_rgb(int width, int height, std::uint32_t seed, float lo = 0.0F, float hi = 1.0F);

// Une image décodée en mémoire hôte, ses lignes séparées de pad octets de bourrage.
DecodedFrame host_frame(const kernels::YuvImage& image, const ColorInfo& color, std::uint32_t id, int pad = 12);
PixelFormat pixel_format(kernels::Format format);

float max_abs_diff(const kernels::RgbImage& a, const kernels::RgbImage& b);

// ModelProto écrit dans dir/name.onnx, avec un ModelSpec qui le décrit.
ModelSpec write_model(const std::filesystem::path& dir, std::string_view name, std::span<const std::byte> bytes,
                      TensorType precision);

}  // namespace ajp::gpu::vk::test
