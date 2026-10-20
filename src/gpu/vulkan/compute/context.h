// Contexte de calcul Vulkan : un appareil, sa file compute et ses tables de fonctions.
// Deux modes. Adopter l'appareil d'un autre (libplacebo en lecture) : le contexte ne détruit
// alors que ce qu'il a créé lui-même. Créer son propre appareil (tests, outils, banc).
// Un contexte appartient au thread qui l'utilise ; il ne se copie ni ne se déplace.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "core/types.h"
#include "gpu/vulkan/compute/caps.h"
#include "gpu/vulkan/compute/functions.h"
#include "gpu/vulkan/compute/library.h"
#include "gpu/vulkan/compute/refuse.h"
#include "gpu/vulkan/compute/vulkan.h"

namespace spdlog {
class logger;
}  // namespace spdlog

namespace ajp::gpu::vk {

class Recording;

// Un appareil à créer.
struct CreateOptions {
  // Chargeur : ce vkGetInstanceProcAddr s'il est donné, sinon la bibliothèque library, sinon
  // le chargeur du système.
  PFN_vkGetInstanceProcAddr get_instance_proc_addr = nullptr;
  std::optional<std::filesystem::path> library;
  std::optional<DeviceUuid> device;     // choisi par UUID ; par défaut le premier énuméré
  std::vector<std::string> extensions;  // extensions d'appareil exigées en plus
  bool validation = false;              // VK_LAYER_KHRONOS_validation (--vulkan-validation)
  // VK_KHR_cooperative_matrix et le modèle mémoire Vulkan, activés si l'appareil les a tous
  // deux ; caps().enabled.cooperative_matrix dit s'ils le sont.
  bool cooperative_matrix = false;
};

// Un appareil existant. Les champs reprennent ceux de pl_vulkan.
struct AdoptedDevice {
  PFN_vkGetInstanceProcAddr get_instance_proc_addr = nullptr;
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  std::uint32_t api_version = 0;  // VkApplicationInfo.apiVersion de l'instance
  std::uint32_t queue_family = 0;
  std::uint32_t queue_index = 0;
  const VkPhysicalDeviceFeatures2* features = nullptr;  // la chaîne passée à vkCreateDevice
  std::vector<std::string> extensions;                  // extensions activées sur le VkDevice
  // Accès exclusif à la file partagée (pl_vulkan.lock_queue et unlock_queue) ; vides quand la
  // file n'est qu'à nous.
  std::function<void()> lock_queue;
  std::function<void()> unlock_queue;
};

// Une valeur d'un sémaphore timeline : ce qu'une soumission attend ou signale.
struct TimelinePoint {
  VkSemaphore semaphore = VK_NULL_HANDLE;
  std::uint64_t value = 0;
};

class VkContext {
 public:
  // Refus : LibraryUnavailable (chargeur ou fonction absents), DeviceUnavailable (aucun appareil,
  // UUID introuvable, aucune file compute), FeatureUnavailable (Vulkan < 1.2, timeline
  // semaphores, extension ou couche de validation absentes), VulkanCallFailed.
  static std::expected<std::unique_ptr<VkContext>, Refusal> create(const CreateOptions& options = {});
  static std::expected<std::unique_ptr<VkContext>, Refusal> adopt(AdoptedDevice adopted);

  VkContext(const VkContext&) = delete;
  VkContext& operator=(const VkContext&) = delete;
  VkContext(VkContext&&) = delete;
  VkContext& operator=(VkContext&&) = delete;
  ~VkContext();

  const InstanceFunctions& instance_functions() const { return instance_fn_; }
  const DeviceFunctions& fn() const { return device_fn_; }
  PFN_vkGetInstanceProcAddr get_instance_proc_addr() const { return get_instance_proc_addr_; }
  VkInstance instance() const { return instance_; }
  VkPhysicalDevice physical_device() const { return physical_device_; }
  VkDevice device() const { return device_; }
  VkQueue queue() const { return queue_; }
  std::uint32_t queue_family() const { return queue_family_; }
  std::uint32_t queue_index() const { return queue_index_; }
  const DeviceCaps& caps() const { return caps_; }
  const DeviceUuid& uuid() const { return caps_.uuid; }
  bool owns_device() const { return owns_device_; }
  spdlog::logger& log() const { return *log_; }

  // Soumet l'enregistrement sur la file compute : il attend waits côté GPU et signale
  // signals. Ses exécutions successives s'enchaînent côté GPU, sans attente hôte.
  Done submit(Recording& recording, std::span<const TimelinePoint> waits = {},
              std::span<const TimelinePoint> signals = {});

  // Comme submit, mais l'enregistrement rejoint un lot que flush() soumet d'un seul
  // vkQueueSubmit (une soumission par image plutôt qu'une par noyau). submit, run_once,
  // wait_on_host et la destruction d'un enregistrement vident le lot d'abord : l'ordre des
  // soumissions reste celui des appels, et aucune attente ne porte sur un travail jamais soumis.
  // Un sémaphore lu hors du contexte (autre file, autre API) n'est fiable qu'après flush().
  Done enqueue(Recording& recording, std::span<const TimelinePoint> waits = {},
               std::span<const TimelinePoint> signals = {});
  Done flush();
  std::uint64_t queue_submits() const { return queue_submits_; }  // appels à vkQueueSubmit

  // Attente hôte d'une valeur (tests, outils, envoi et lecture hôte) ; chaque appel compte dans
  // host_waits. Refus VulkanCallFailed au délai dépassé ou à la perte de l'appareil.
  Done wait_on_host(const TimelinePoint& point, std::chrono::nanoseconds timeout = std::chrono::seconds(10));
  std::uint64_t host_waits() const { return host_waits_; }

  // Enregistre, soumet et attend sur l'hôte un command buffer jetable : envoi et lecture hôte,
  // transitions d'images à la création. Compte une attente hôte.
  Done run_once(const std::function<void(VkCommandBuffer)>& record);

 private:
  VkContext() = default;
  Done create_instance(const CreateOptions& options);
  Done create_device(const CreateOptions& options);
  Done finish_setup();  // file, outils internes, journal
  Done queue_submit(std::span<const VkSubmitInfo> infos);

  // Une soumission du lot, ses tableaux réutilisés d'un lot à l'autre.
  struct Batched {
    Recording* recording = nullptr;
    std::vector<VkSemaphore> wait_semaphores;
    std::vector<std::uint64_t> wait_values;
    std::vector<VkPipelineStageFlags> stages;
    std::vector<VkSemaphore> signal_semaphores;
    std::vector<std::uint64_t> signal_values;
  };

  std::optional<VulkanLibrary> library_;
  PFN_vkGetInstanceProcAddr get_instance_proc_addr_ = nullptr;
  InstanceFunctions instance_fn_;
  DeviceFunctions device_fn_;
  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  std::uint32_t queue_family_ = 0;
  std::uint32_t queue_index_ = 0;
  bool owns_instance_ = false;
  bool owns_device_ = false;
  std::function<void()> lock_queue_;
  std::function<void()> unlock_queue_;
  DeviceCaps caps_;
  // Outils de run_once : un pool jetable et un sémaphore qui compte ses exécutions.
  VkCommandPool once_pool_ = VK_NULL_HANDLE;
  VkSemaphore once_done_ = VK_NULL_HANDLE;
  std::uint64_t once_count_ = 0;
  std::uint64_t host_waits_ = 0;
  std::uint64_t queue_submits_ = 0;
  std::vector<Batched> batch_;
  std::size_t batched_ = 0;  // entrées de batch_ en attente
  std::vector<VkTimelineSemaphoreSubmitInfo> batch_timelines_;
  std::vector<VkSubmitInfo> batch_infos_;
  std::vector<VkCommandBuffer> batch_buffers_;
  std::shared_ptr<spdlog::logger> log_;
};

}  // namespace ajp::gpu::vk
