// Outils partagés par vulkan_backend.cpp et vulkan_backend_runtime.cpp.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

#include "core/types.h"
#include "gpu/tensor.h"

namespace ajp::gpu::vk {

// Projections hôte des tenseurs alloués pour un runtime (MemoryUse::Readback), par handle de
// tampon : enqueue les retrouve à partir des Tensor. Partagées entre le backend et ses tampons,
// qui s'effacent en mourant.
struct HostViews {
  struct View {
    std::byte* data = nullptr;
    std::uint64_t bytes = 0;
  };
  std::mutex mutex;
  std::unordered_map<std::uintptr_t, View> views;
};

namespace detail {

std::unexpected<Refusal> refuse_backend(RefusalCode code, std::string text);
std::size_t element_bytes(TensorType type);
// Octets denses d'un tenseur NCHW ; refus FormatUnsupported, ShapeInvalid (dimensions hors de
// [1, 2^24], plus de 2^32 éléments).
std::expected<std::uint64_t, Refusal> tensor_bytes(TensorType type, const Shape& shape);
// Nom de fichier sûr pour l'identifiant d'un modèle.
std::string file_stem(std::string_view id);

}  // namespace detail
}  // namespace ajp::gpu::vk
