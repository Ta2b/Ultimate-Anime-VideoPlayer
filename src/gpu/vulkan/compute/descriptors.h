// Ensembles de descripteurs d'un enregistrement : un par dispatch, tirés de pools aux bornes de
// sa création. Un pool plein en ouvre un autre, assez grand pour l'ensemble demandé : la taille
// d'un enregistrement (dix splats d'un instant de GMFSS, un ingest réduit et ses vues) ne le fait
// pas échouer. Au-delà de kMaxPools pools, la demande est refusée (GpuMemoryExhausted) avant
// d'atteindre le pilote ; rien n'est jamais réécrit après l'enregistrement, qui peut être rejoué
// tel quel.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "core/types.h"
#include "gpu/vulkan/compute/buffer.h"
#include "gpu/vulkan/compute/context.h"
#include "gpu/vulkan/compute/pipeline.h"
#include "gpu/vulkan/compute/refuse.h"
#include "gpu/vulkan/compute/vulkan.h"

namespace ajp::gpu::vk {

struct DescriptorLimits {
  std::uint32_t sets = 64;      // dispatchs par enregistrement
  std::uint32_t buffers = 512;  // tampons de stockage, tous ensembles confondus
  std::uint32_t images = 16;    // images, de stockage ou échantillonnées
};

class DescriptorArena {
 public:
  static constexpr std::size_t kMaxPools = 16;

  // Refus : FieldOutOfRange (borne nulle), VulkanCallFailed.
  static std::expected<DescriptorArena, Refusal> create(const VkContext& context, DescriptorLimits limits);

  DescriptorArena(DescriptorArena&& other) noexcept;
  DescriptorArena& operator=(DescriptorArena&& other) noexcept;
  DescriptorArena(const DescriptorArena&) = delete;
  DescriptorArena& operator=(const DescriptorArena&) = delete;
  ~DescriptorArena();

  // Un ensemble pour ce pipeline, ses tampons aux bindings 0..n−1 puis ses images (vues en
  // VK_IMAGE_LAYOUT_GENERAL) : celles de stockage (pipeline.images()), puis les échantillonnées
  // (pipeline.sampled(), avec l'échantillonneur immuable du pipeline). Refus : GpuMemoryExhausted
  // (kMaxPools pools pleins), VulkanCallFailed.
  std::expected<VkDescriptorSet, Refusal> bind(const Pipeline& pipeline, std::span<const BufferView> buffers,
                                               std::span<const VkImageView> images);

  std::uint32_t sets_used() const { return total_sets_; }
  std::size_t pools() const { return pools_.size(); }

 private:
  DescriptorArena(const VkContext* context, DescriptorLimits limits) : context_(context), limits_(limits) {}
  // Un pool neuf, aux bornes de l'arène et au moins à celles d'un ensemble de buffers tampons
  // et images images ; il devient le courant.
  Done open_pool(std::uint32_t buffers, std::uint32_t images);
  void destroy();

  const VkContext* context_ = nullptr;
  std::vector<VkDescriptorPool> pools_;
  DescriptorLimits limits_;
  DescriptorLimits current_;  // les bornes du pool courant, le dernier
  std::uint32_t sets_ = 0;    // tirés du pool courant
  std::uint32_t buffers_ = 0;
  std::uint32_t images_ = 0;
  std::uint32_t total_sets_ = 0;
};

}  // namespace ajp::gpu::vk
