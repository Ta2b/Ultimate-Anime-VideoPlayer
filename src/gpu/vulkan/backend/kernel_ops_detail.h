// Outils internes des noyaux de la chaîne (kernel_ops.cpp, kernel_frames.cpp, kernel_ring.cpp).
#pragma once

#include <algorithm>
#include <cstdint>
#include <expected>
#include <string_view>
#include <type_traits>
#include <utility>

#include "core/frame.h"
#include "gpu/tensor.h"
#include "gpu/vulkan/backend/kernel_ops.h"
#include "gpu/vulkan/compute/destroyed_log.h"
#include "gpu/vulkan/compute/kernel_types.h"

namespace ajp::gpu::vk {

namespace detail {

// Handle Vulkan non dispatchable vers un entier et retour : un pointeur en 64 bits, un entier
// de 64 bits ailleurs.
template <class Handle>
Handle from_bits(std::uint64_t bits) {
  if constexpr (std::is_pointer_v<Handle>) {
    return reinterpret_cast<Handle>(static_cast<std::uintptr_t>(bits));
  } else {
    return static_cast<Handle>(bits);
  }
}

template <class Handle>
std::uint64_t to_bits(Handle handle) {
  return handle_word(handle);
}

VkBuffer to_buffer(std::uintptr_t handle);
std::uint64_t handle_bits(VkBuffer buffer);
// Un refus du socle dans le vocabulaire de KernelOps (InputMismatch devient ShapeInvalid).
std::unexpected<Refusal> contract_refusal(Refusal refusal);
// Tenseur 1 × planes × h × w (planes 0 : quelconque) vu par les noyaux du socle.
std::expected<TensorView, Refusal> tensor_view(const Tensor& t, std::uint32_t planes, std::string_view what);
void add_view(OpKey& key, const TensorView& v);

// Format et couleur d'une image décodée dans le vocabulaire des noyaux. Refus FormatUnsupported :
// format RGB, matrice RGB.
std::expected<kernels::Format, Refusal> kernel_format(PixelFormat format);
std::expected<ColorSpec, Refusal> color_spec(const ColorInfo& color);
VkImage to_image(std::uintptr_t handle);
VkSemaphore to_semaphore(std::uintptr_t handle);
std::uint64_t image_bits(VkImage image);

// Un plan d'une image YUV : texels, lignes, octets par texel, et sa place dans la disposition
// serrée du tampon.
struct PlaneCopy {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t texel = 0;
  PlaneLayout dst;
};
// Deux plans (chroma entrelacée) ou trois.
std::uint32_t plane_copies(const YuvView& v, std::array<PlaneCopy, 3>& out);

// Ferme les accès Vulkan ouverts par begin_vulkan_access : release_all après la soumission,
// ou, à la destruction, sans soumission (une erreur entre l'ouverture et vkQueueSubmit).
class AccessGuard {
 public:
  AccessGuard() = default;
  AccessGuard(const AccessGuard&) = delete;
  AccessGuard& operator=(const AccessGuard&) = delete;
  AccessGuard(AccessGuard&&) = delete;
  AccessGuard& operator=(AccessGuard&&) = delete;
  ~AccessGuard() { release_all({}, false); }

  void hold(const FrameData* frame, const VulkanImageState& state) {
    frames_[count_] = frame;
    states_[count_] = state;
    ++count_;
  }
  // La responsabilité passe à un autre garde (submit_frames).
  void dismiss() { count_ = 0; }
  void release_all(const VulkanImageState& after, bool submitted) {
    for (std::size_t k = 0; k < count_; ++k)
      frames_[k]->end_vulkan_access(submitted ? after : states_[k], submitted);
    count_ = 0;
  }

 private:
  std::array<const FrameData*, 2> frames_{};
  std::array<VulkanImageState, 2> states_{};
  std::size_t count_ = 0;
};

}  // namespace detail

template <class Body>
std::expected<Recording*, Refusal> VulkanKernelOps::recording(const OpKey& key, Body&& body) {
  if (!key.valid())
    return std::unexpected(Refusal{RefusalCode::InputMismatch, "clé d'enregistrement trop longue"});
  ++uses_;
  forget_destroyed();
  if (auto found = std::ranges::find(recordings_, key, &Cached::key); found != recordings_.end()) {
    found->used = uses_;
    return found->recording.get();
  }
  // Assez pour les dix splats d'un instant déformé après SPAN : deux dispatchs chacun.
  auto recorder = Recorder::begin(context_, memory_, {.sets = 32, .buffers = 128, .images = 4});
  if (!recorder) return std::unexpected(recorder.error());
  if (auto ok = body(*recorder); !ok) return detail::contract_refusal(ok.error());
  auto made = recorder->finish();
  if (!made) return std::unexpected(made.error());
  if (recordings_.size() >= kMaxRecordings) {
    // Le plus ancien part ; sa destruction attend sa dernière exécution (rare : clés bornées).
    recordings_.erase(std::ranges::min_element(recordings_, {}, &Cached::used));
  }
  recordings_.push_back({.key = key, .recording = std::move(*made), .used = uses_});
  return recordings_.back().recording.get();
}

}  // namespace ajp::gpu::vk
