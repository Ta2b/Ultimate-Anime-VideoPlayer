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
