// Chronométrage GPU d'un Recording par requêtes timestamp, sans toucher à l'enregistrement :
//   porte     : les trois soumissions attendent que l'hôte les ait toutes mises en file
//               (vkSignalSemaphore), pour que sa latence ne tombe pas dans la mesure ;
//   ouverture : remplissage facultatif d'un tampon plus grand que le cache L2 (le cache se
//               retrouve plein de lignes sales, comme après l'étape précédente de la chaîne),
//               remise à zéro des deux requêtes, timestamp 0, puis une barrière ALL_COMMANDS
//               qui retient tout ce qui suit dans l'ordre de soumission ;
//   l'enregistrement, soumis par VkContext::submit, sans sémaphore à attendre après le
//               timestamp 0 (la porte est ouverte avant même l'ouverture) ;
//   clôture : timestamp 1 en BOTTOM_OF_PIPE, donc après toute commande soumise avant lui.
// L'hôte attend la clôture puis lit les deux valeurs : une exécution à la fois, GPU au repos
// entre deux. La durée inclut les frontières entre lots et la barrière finale de
// l'enregistrement ; le cas « plancher » du banc (un remplissage de 4 octets) les mesure seules.
// Le contexte doit posséder sa file (VkContext::create) : l'ouverture et la clôture la
// soumettent directement, sans le verrou d'une file adoptée.
#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <optional>

#include "gpu/vulkan/compute/buffer.h"
#include "gpu/vulkan/compute/context.h"
#include "gpu/vulkan/compute/memory.h"
#include "gpu/vulkan/compute/recording.h"
#include "gpu/vulkan/compute/refuse.h"
#include "gpu/vulkan/compute/sync.h"

namespace ajp::bench {

struct QueryFunctions {
  PFN_vkCreateQueryPool create_query_pool = nullptr;
  PFN_vkDestroyQueryPool destroy_query_pool = nullptr;
  PFN_vkCmdResetQueryPool cmd_reset_query_pool = nullptr;
  PFN_vkCmdWriteTimestamp cmd_write_timestamp = nullptr;
  PFN_vkGetQueryPoolResults get_query_pool_results = nullptr;
};

class GpuTimer {
 public:
  // flush_bytes : taille du tampon de vidage du L2, 0 pour garder le cache chaud.
  // Refus : DeviceUnavailable (contexte adopté), LibraryUnavailable (fonction de requête
  // absente), FeatureUnavailable (file sans timestamps), ceux de Buffer::create et de
  // TimelineSemaphore::create, VulkanCallFailed.
  static std::expected<std::unique_ptr<GpuTimer>, Refusal> create(gpu::vk::VkContext& context,
                                                                  gpu::vk::MemoryAllocator& memory,
                                                                  VkDeviceSize flush_bytes);

  GpuTimer(const GpuTimer&) = delete;
  GpuTimer& operator=(const GpuTimer&) = delete;
  GpuTimer(GpuTimer&&) = delete;
  GpuTimer& operator=(GpuTimer&&) = delete;
  // Attend la dernière clôture soumise avant de libérer quoi que ce soit.
  ~GpuTimer();

  // Une exécution chronométrée de recording, en millisecondes GPU. Refus : ceux de
  // VkContext::submit et wait_on_host, VulkanCallFailed.
  std::expected<double, Refusal> time(gpu::vk::Recording& recording);

  std::uint32_t valid_bits() const { return valid_bits_; }
  double period_ns() const { return period_ns_; }
  VkDeviceSize flush_bytes() const { return flush_ ? flush_->size() : 0; }

 private:
  GpuTimer(gpu::vk::VkContext& context, gpu::vk::TimelineSemaphore done)
      : context_(context), done_(std::move(done)) {}
  gpu::vk::Done record_brackets();
  gpu::vk::Done submit(VkCommandBuffer command_buffer, std::optional<std::uint64_t> wait_value,
                       std::optional<std::uint64_t> signal_value);

  gpu::vk::VkContext& context_;
  gpu::vk::TimelineSemaphore done_;  // porte (hôte) puis clôture : valeurs croissantes
  QueryFunctions query_;
  std::optional<gpu::vk::Buffer> flush_;
  VkQueryPool pool_ = VK_NULL_HANDLE;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
  VkCommandBuffer open_ = VK_NULL_HANDLE;
  VkCommandBuffer close_ = VK_NULL_HANDLE;
  std::uint64_t signaled_ = 0;  // dernière valeur de done_ promise par une soumission réussie
  std::uint32_t valid_bits_ = 0;
  double period_ns_ = 0.0;
};

}  // namespace ajp::bench
