// GpuDevice : l'appareil de rendu tel que present l'a créé, décrit pour les modules qui le
// partagent (media pour le décodage Vulkan, gpu et les backends pour l'interop).
// Fait partie de core_types : bibliothèque standard seulement, les handles Vulkan restent
// opaques et chaque module les reconvertit vers les types de vulkan.h.
//
// Les champs Vulkan suivent ce que libplacebo (pl_vulkan, pl_vulkan_import_params) et FFmpeg
// (AVVulkanDeviceContext) demandent pour partager un VkDevice : instance, appareil physique,
// appareil logique, vkGetInstanceProcAddr, version d'API, extensions et fonctionnalités
// activées, familles de files et leur verrou.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ajp {

enum class GpuApi : std::uint8_t { Vulkan, Cuda, D3d11, Metal };
enum class GpuVendor : std::uint8_t { Nvidia, Amd, Intel, Apple, Other };

// Identifiants PCI des fabricants (VkPhysicalDeviceProperties::vendorID, DXGI_ADAPTER_DESC).
namespace pci {
inline constexpr std::uint32_t kNvidia = 0x10DE;
inline constexpr std::uint32_t kAmd = 0x1002;
inline constexpr std::uint32_t kIntel = 0x8086;
inline constexpr std::uint32_t kApple = 0x106B;
}  // namespace pci

constexpr GpuVendor vendor_from_pci(std::uint32_t vendor_id) {
  switch (vendor_id) {
    case pci::kNvidia:
      return GpuVendor::Nvidia;
    case pci::kAmd:
      return GpuVendor::Amd;
    case pci::kIntel:
      return GpuVendor::Intel;
    case pci::kApple:
      return GpuVendor::Apple;
    default:
      return GpuVendor::Other;
  }
}

// VkPhysicalDeviceIDProperties::deviceUUID, cuDeviceGetUuid, pl_gpu.uuid : le même octet à
// octet pour un même GPU, ce qui relie l'appareil CUDA ou D3D11 à l'appareil Vulkan.
using DeviceUuid = std::array<std::uint8_t, 16>;
// Windows : LUID de l'adaptateur (VkPhysicalDeviceIDProperties::deviceLUID, DXGI).
using DeviceLuid = std::array<std::uint8_t, 8>;

// Pointeur de fonction générique : PFN_vkGetInstanceProcAddr y est rangé par reinterpret_cast
// et doit être reconverti vers son type exact avant l'appel (aller-retour défini par la norme).
using VoidFunction = void (*)();

// Verrou d'une file, à la manière de pl_vulkan_import_params::lock_queue et
// d'AVVulkanDeviceContext::lock_queue : toute soumission sur la file (family, index), quel
// que soit le module, se fait entre lock_queue et unlock_queue, avec le même context.
using QueueLock = void (*)(void* context, std::uint32_t family, std::uint32_t index);

// VkQueueFlagBits, recopiés pour rester sans vulkan.h.
namespace queue_flags {
inline constexpr std::uint32_t kGraphics = 0x1;
inline constexpr std::uint32_t kCompute = 0x2;
inline constexpr std::uint32_t kTransfer = 0x4;
inline constexpr std::uint32_t kVideoDecode = 0x20;  // VK_QUEUE_VIDEO_DECODE_BIT_KHR
}  // namespace queue_flags

struct QueueFamily {
  std::uint32_t index = 0;       // indice de la famille dans l'appareil physique
  std::uint32_t count = 0;       // files créées dans cette famille
  std::uint32_t flags = 0;       // VkQueueFlags
  std::uint32_t video_caps = 0;  // VkVideoCodecOperationFlagsKHR ; 0 sans décodage vidéo
};

// Une file précise : sa famille et son rang dans la famille.
struct QueueRef {
  std::uint32_t family = 0;
  std::uint32_t index = 0;
};

struct VulkanDevice {
  void* instance = nullptr;                       // VkInstance
  void* physical_device = nullptr;                // VkPhysicalDevice
  void* device = nullptr;                         // VkDevice
  VoidFunction get_instance_proc_addr = nullptr;  // PFN_vkGetInstanceProcAddr
  std::uint32_t api_version = 0;                  // VK_MAKE_API_VERSION de l'instance
  std::vector<std::string> instance_extensions;   // activées à la création
  std::vector<std::string> device_extensions;
  // Chaîne VkPhysicalDeviceFeatures2 activée, possédée par present le temps de l'appareil.
  const void* enabled_features = nullptr;
  std::vector<QueueFamily> families;  // seulement celles où des files ont été créées
  QueueRef graphics;
  QueueRef compute;
  std::optional<QueueRef> video_decode;  // absente : pas de décodage Vulkan partagé
  QueueLock lock_queue = nullptr;        // nullptr : files non partagées, aucun verrou
  QueueLock unlock_queue = nullptr;
  void* queue_context = nullptr;  // premier argument de lock_queue et unlock_queue
};

struct GpuDevice {
  GpuApi api = GpuApi::Vulkan;
  std::uint32_t pci_vendor = 0;  // vendorID ; le fabricant s'en déduit, jamais du nom
  std::uint32_t pci_device = 0;  // deviceID
  std::string name;              // « NVIDIA GeForce RTX 4070 Ti »
  DeviceUuid uuid{};
  std::optional<DeviceLuid> luid;      // Windows seulement
  std::string driver_version;          // lisible : « 580.82.07 », « Mesa 25.1.4 »
  std::optional<VulkanDevice> vulkan;  // présent si et seulement si api vaut Vulkan

  GpuVendor vendor() const { return vendor_from_pci(pci_vendor); }
};

}  // namespace ajp
