// Contexte de calcul Vulkan : un appareil, sa file compute et ses tables de fonctions.
// Deux modes. Adopter l'appareil d'un autre (libplacebo en lecture) : le contexte ne détruit
// alors que ce qu'il a créé lui-même. Créer son propre appareil (tests, outils, banc).
// Un contexte appartient au thread qui l'utilise ; il ne se copie ni ne se déplace.
#pragma once

#include <atomic>
#include <chrono>
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
  // Messages du chargeur, des couches et du pilote (VK_EXT_debug_utils), journalisés et comptés
  // (validation_errors) ; implicite avec validation.
  bool debug_messages = false;
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
  // semaphores, extension, VK_EXT_debug_utils ou couche de validation absentes), VulkanCallFailed.
  static std::expected<std::unique_ptr<VkContext>, Refusal> create(const CreateOptions& options = {});
  // Refus en plus : InputMismatch (lock_queue sans unlock_queue, ou l'inverse). Tout ce que le
  // contexte crée (Recording, MemoryAllocator, PipelineCache…) doit disparaître avant lui.
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

  // Messages VK_EXT_debug_utils (CreateOptions.validation ou debug_messages) : actifs, et nombre
  // d'erreurs et d'avertissements reçus depuis la création. Toujours 0 sans messager.
  bool debug_messages() const { return messenger_ != VK_NULL_HANDLE; }
  std::uint64_t validation_errors() const { return debug_ ? debug_->errors.load() : 0; }
  std::uint64_t validation_warnings() const { return debug_ ? debug_->warnings.load() : 0; }

  // Soumet l'enregistrement sur la file compute : il attend waits côté GPU et signale
  // signals. Ses exécutions successives s'enchaînent côté GPU, sans attente hôte. Refus :
  // InputMismatch (enregistrement d'un autre contexte ou non terminé, sémaphore nul, sémaphore
  // runs de l'enregistrement attendu ou signalé, sémaphore signalé deux fois), FieldOutOfRange
  // (signal d'une valeur déjà atteinte), VulkanCallFailed.
  Done submit(Recording& recording, std::span<const TimelinePoint> waits = {},
              std::span<const TimelinePoint> signals = {});

  // Attente hôte d'une valeur (tests, outils, envoi et lecture hôte) ; chaque appel compte dans
  // host_waits. Refus VulkanCallFailed au délai dépassé ou à la perte de l'appareil.
  Done wait_on_host(const TimelinePoint& point, std::chrono::nanoseconds timeout = std::chrono::seconds(10));
  std::uint64_t host_waits() const { return host_waits_; }

  // Enregistre, soumet et attend sur l'hôte un command buffer jetable : envoi et lecture hôte,
  // transitions d'images à la création. Compte une attente hôte.
  Done run_once(const std::function<void(VkCommandBuffer)>& record);

 private:
  // Compteurs du messager, appelé depuis n'importe quel thread du pilote : atomiques.
  struct DebugSink {
    std::atomic<std::uint64_t> errors{0};
    std::atomic<std::uint64_t> warnings{0};
    std::shared_ptr<spdlog::logger> log;
  };
  // Attente bornée des run_once encore en vol à la destruction d'un contexte adopté.
  static constexpr std::chrono::seconds kTeardownTimeout{10};

  VkContext() = default;
  Done create_instance(const CreateOptions& options);
  Done create_device(const CreateOptions& options);
  Done finish_setup();  // file, outils internes, journal
  Done queue_submit(const VkSubmitInfo& info);
  Done check_signal(const TimelinePoint& point, std::span<const VkSemaphore> batch) const;
  bool wait_once_idle() const;
  // Messager VK_EXT_debug_utils (debug.cpp) : description chaînée à vkCreateInstance, puis
  // messager propre une fois l'instance créée.
  VkDebugUtilsMessengerCreateInfoEXT messenger_info() const;
  Done create_messenger();
  void destroy_messenger();
  static VKAPI_ATTR VkBool32 VKAPI_CALL on_debug_message(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                         VkDebugUtilsMessageTypeFlagsEXT types,
                                                         const VkDebugUtilsMessengerCallbackDataEXT* data,
                                                         void* user);

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
  std::uint64_t once_count_ = 0;  // soumissions de run_once acceptées par la file
  std::uint64_t host_waits_ = 0;
  std::shared_ptr<spdlog::logger> log_;
  std::unique_ptr<DebugSink> debug_;
  VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
  PFN_vkDestroyDebugUtilsMessengerEXT destroy_messenger_ = nullptr;
};

}  // namespace ajp::gpu::vk
