// Tenseurs résidents sur l'appareil d'un backend : ce que moteurs (backend.h) et noyaux
// (kernel_ops.h) lisent et écrivent.
#pragma once

#include <array>
#include <cstdint>

namespace ajp::gpu {

enum class TensorType : std::uint8_t { F32, F16, U8, I8 };

// Forme NCHW.
struct Shape {
  std::array<std::int64_t, 4> dims{};
  friend bool operator==(const Shape&, const Shape&) = default;
};

// La taille dense d'un tenseur, la même borne pour tous les backends (P2.28) : chaque dimension dans
// [1, 2^24], au plus 2^32 éléments. 0 : forme refusée ou type inconnu.
inline constexpr std::int64_t kMaxTensorDim = std::int64_t{1} << 24;
inline constexpr std::uint64_t kMaxTensorElements = std::uint64_t{1} << 32;
constexpr std::uint64_t tensor_element_bytes(TensorType type) {
  switch (type) {
    case TensorType::F32:
      return 4;
    case TensorType::F16:
      return 2;
    case TensorType::U8:
    case TensorType::I8:
      return 1;
  }
  return 0;
}
constexpr std::uint64_t dense_tensor_bytes(TensorType type, const Shape& shape) {
  std::uint64_t count = 1;
  for (const std::int64_t d : shape.dims) {
    if (d <= 0 || d > kMaxTensorDim) return 0;
    count *= static_cast<std::uint64_t>(d);
    if (count > kMaxTensorElements) return 0;
  }
  return count * tensor_element_bytes(type);
}

// Tenseur résident sur l'appareil du backend. Ne possède pas forcément sa mémoire.
struct Tensor {
  TensorType type = TensorType::F16;
  Shape shape;
  std::uintptr_t handle = 0;  // CUdeviceptr, VkBuffer…
  std::uint64_t offset = 0;
};

// Mémoire d'un tenseur allouée par Backend::alloc_tensor, libérée à sa destruction ; elle
// doit être détruite avant le Backend qui l'a allouée.
class TensorBuffer {
 public:
  virtual ~TensorBuffer() = default;
  virtual const Tensor& tensor() const = 0;
};

}  // namespace ajp::gpu
