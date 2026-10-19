// Enregistrement d'un command buffer réutilisable : écrit une fois par un Recorder (dispatchs,
// remplissages, barrières entre eux), puis rejoué à chaque soumission (VkContext::submit).
// Chaque exécution signale la valeur suivante du sémaphore timeline de l'enregistrement, et
// la suivante l'attend côté GPU : deux exécutions ne se chevauchent jamais, sans attente hôte.
// L'enregistrement possède ses tampons de travail et de constantes ; les tampons et images
// liés par l'appelant doivent vivre aussi longtemps que lui.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "core/types.h"
#include "gpu/vulkan/compute/buffer.h"
#include "gpu/vulkan/compute/context.h"
#include "gpu/vulkan/compute/descriptors.h"
#include "gpu/vulkan/compute/memory.h"
#include "gpu/vulkan/compute/pipeline.h"
#include "gpu/vulkan/compute/refuse.h"
#include "gpu/vulkan/compute/sync.h"
#include "gpu/vulkan/compute/vulkan.h"

namespace ajp::gpu::vk {

class Recording {
 public:
  Recording(const Recording&) = delete;
  Recording& operator=(const Recording&) = delete;
  Recording(Recording&&) = delete;
  Recording& operator=(Recording&&) = delete;
  // Attend la fin de la dernière exécution soumise avant de libérer quoi que ce soit.
  ~Recording();

  VkCommandBuffer command_buffer() const { return command_buffer_; }
  const VkContext& context() const { return context_; }
  bool finished() const { return finished_; }
  std::uint32_t dispatches() const { return dispatches_; }
  std::uint64_t submitted() const { return submitted_; }  // exécutions soumises
  // Atteint quand la dernière exécution soumise s'achève ; valeur 0 avant la première.
  TimelinePoint last_run() const { return runs_.at(submitted_); }
  const TimelineSemaphore& runs() const { return runs_; }

 private:
  friend class Recorder;
  friend class VkContext;
  Recording(VkContext& context, TimelineSemaphore runs, DescriptorArena arena)
      : context_(context), runs_(std::move(runs)), arena_(std::move(arena)) {}

  VkContext& context_;
  TimelineSemaphore runs_;
  DescriptorArena arena_;
  VkCommandPool pool_ = VK_NULL_HANDLE;
  VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
  std::vector<Buffer> owned_;
  std::uint32_t dispatches_ = 0;
  std::uint64_t submitted_ = 0;
  bool finished_ = false;
};

class Recorder {
 public:
  // Refus : ceux de DescriptorArena::create, VulkanCallFailed.
  static std::expected<Recorder, Refusal> begin(VkContext& context, MemoryAllocator& memory,
                                                DescriptorLimits limits = {});

  VkContext& context() const { return *context_; }
  MemoryAllocator& memory() const { return *memory_; }

  // Tampon de travail possédé par l'enregistrement.
  std::expected<BufferView, Refusal> scratch(VkDeviceSize bytes, MemoryUse use = MemoryUse::Device);
  // Données de l'hôte figées dans l'enregistrement (tables de poids) : un tampon Upload rempli
  // maintenant, lu à chaque exécution.
  std::expected<BufferView, Refusal> constants(std::span<const std::byte> bytes);

  // Un dispatch, après une barrière sur tout ce qui le précède dans l'enregistrement. Refus :
  // InputMismatch (nombre de tampons, d'images ou d'octets de push différent du pipeline ; vue
  // vide, mal alignée ou trop grande), FeatureUnavailable (groupes au-delà des limites),
  // GpuMemoryExhausted (pool de descripteurs plein).
  Done dispatch(const Pipeline& pipeline, std::span<const BufferView> buffers,
                std::span<const VkImageView> images, std::span<const std::byte> push,
                std::array<std::uint32_t, 3> groups);

  // Remplit la vue de value à chaque exécution (offset et taille multiples de 4).
  Done fill(const BufferView& view, std::uint32_t value);

  // Commandes écrites par l'appelant avec ses propres fonctions (copies, horodatages), après
  // une barrière sur tout ce qui précède ; ce qui suit les attend aussi. Refus InputMismatch :
  // déjà terminé.
  Done commands(const std::function<void(VkCommandBuffer)>& record);

  // Barrière finale vers l'hôte et les transferts, puis vkEndCommandBuffer. Le Recorder est
  // ensuite vide. Refus InputMismatch : déjà terminé.
  std::expected<std::unique_ptr<Recording>, Refusal> finish();

 private:
  Recorder(VkContext& context, MemoryAllocator& memory, std::unique_ptr<Recording> recording)
      : context_(&context), memory_(&memory), recording_(std::move(recording)) {}
  Done open() const;
  void barrier();
  Done check_view(const BufferView& view, std::string_view what) const;

  VkContext* context_;
  MemoryAllocator* memory_;
  std::unique_ptr<Recording> recording_;
  bool pending_ = false;  // une commande précédente a écrit : barrière avant la suivante
};

}  // namespace ajp::gpu::vk
