// Tenseurs du backend CPU : de la mémoire hôte, dont Tensor::handle est l'adresse. Le registre
// des tampons vivants refuse un tenseur qui sortirait d'eux, au lieu de lire n'importe où.
// Lectures et écritures en plans float de kernels/ref ; F16 s'arrondit comme un port GPU.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <mutex>
#include <span>
#include <string_view>
#include <vector>

#include "core/frame.h"
#include "core/types.h"
#include "gpu/tensor.h"
#include "kernels/image.h"
#include "kernels/ref/color.h"

namespace ajp::gpu::cpu {

std::size_t element_size(TensorType type);
// Éléments d'une forme NCHW ; refus (ShapeInvalid) d'une dimension nulle ou négative, ou d'un
// total au-delà de 2^31 éléments.
std::expected<std::size_t, Refusal> element_count(const Shape& shape);

class HostMemory {
 public:
  void add(std::byte* base, std::size_t size);
  void remove(const std::byte* base);
  // Les octets d'un tenseur, qui doivent tenir dans un tampon vivant (InputMismatch sinon).
  std::expected<std::span<std::byte>, Refusal> bytes(const Tensor& tensor) const;

 private:
  struct Range {
    std::byte* base;
    std::size_t size;
  };
  mutable std::mutex mutex_;  // alloc_tensor peut venir d'un autre thread que les noyaux
  std::vector<Range> ranges_;
};

class HostBuffer final : public TensorBuffer {
 public:
  HostBuffer(HostMemory& memory, TensorType type, const Shape& shape, std::size_t bytes);
  HostBuffer(const HostBuffer&) = delete;
  HostBuffer& operator=(const HostBuffer&) = delete;
  ~HostBuffer() override;
  const Tensor& tensor() const override { return tensor_; }

 private:
  HostMemory& memory_;
  std::vector<std::byte> data_;
  Tensor tensor_;
};

// Un tenseur d'image 1 × channels × height × width en F32 ou F16 ; refus nommés sinon.
std::expected<void, Refusal> expect_image(const Tensor& tensor, int channels, int height, int width,
                                          std::string_view what);
// Dimensions d'un tenseur d'image : (canaux, hauteur, largeur), lot 1, F32 ou F16.
struct ImageShape {
  int channels = 0;
  int height = 0;
  int width = 0;
};
std::expected<ImageShape, Refusal> image_shape(const Tensor& tensor, std::string_view what);

// Toutes les valeurs d'un tenseur F32 ou F16, en float, dans l'ordre NCHW.
std::expected<std::vector<float>, Refusal> read_values(const HostMemory& memory, const Tensor& tensor);
// Exactement autant de valeurs que d'éléments ; F16 s'arrondit au plus proche, égalités au pair.
std::expected<void, Refusal> write_values(const HostMemory& memory, const Tensor& tensor,
                                          std::span<const float> values);

std::expected<std::vector<kernels::PlaneF>, Refusal> read_planes(const HostMemory& memory,
                                                                 const Tensor& tensor);
std::expected<void, Refusal> write_planes(const HostMemory& memory, const Tensor& tensor,
                                          std::span<const kernels::PlaneF> planes);
std::expected<kernels::RgbImage, Refusal> read_rgb(const HostMemory& memory, const Tensor& tensor);
std::expected<void, Refusal> write_rgb(const HostMemory& memory, const Tensor& tensor,
                                       const kernels::RgbImage& image);

// Image décodée en mémoire hôte → YUV de la référence, et sa description de couleur.
std::expected<kernels::YuvImage, Refusal> read_frame(const DecodedFrame& frame);
std::expected<kernels::ref::ColorSpec, Refusal> color_of(const DecodedFrame& frame);

// Refus nommé d'une erreur de la référence : l'opération, puis le code et le détail.
Refusal refusal_of(const kernels::Error& error, std::string_view operation);

}  // namespace ajp::gpu::cpu
