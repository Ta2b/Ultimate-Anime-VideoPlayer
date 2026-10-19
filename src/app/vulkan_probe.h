// Énumération des GPU par Vulkan, sans lier le chargeur : il est ouvert par
// LibraryLoader (libvulkan.so.1, vulkan-1.dll, libvulkan.1.dylib), une instance
// est créée le temps de lire chaque appareil, puis détruite.
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include "app/library_loader.h"
#include "core/engines.h"
#include "core/types.h"

namespace ajp::app {

// Identifiants PCI des constructeurs (VkPhysicalDeviceProperties::vendorID).
namespace pci {
inline constexpr std::uint32_t kNvidia = 0x10DE;
inline constexpr std::uint32_t kAmd = 0x1002;
inline constexpr std::uint32_t kIntel = 0x8086;
inline constexpr std::uint32_t kApple = 0x106B;
}  // namespace pci

enum class DeviceType : std::uint8_t { Discrete, Integrated, Virtual, Cpu, Other };

using DeviceUuid = std::array<std::uint8_t, 16>;

struct VulkanDevice {
  std::string name;             // « NVIDIA GeForce RTX 4070 Ti »
  std::uint32_t vendor_id = 0;  // PCI : pci::kNvidia…
  std::uint32_t device_id = 0;  // PCI
  DeviceType type = DeviceType::Other;
  std::uint32_t api_version = 0;     // VK_MAKE_API_VERSION
  std::uint32_t driver_version = 0;  // brut, codé selon le constructeur
  std::string driver_name;           // VkPhysicalDeviceDriverProperties : « NVIDIA », « radv »
  std::string driver_info;           // « 615.78.08 », « Mesa 25.2.4 »
  DeviceUuid uuid{};                 // deviceUUID : relie Vulkan, CUDA et le décodage
  bool uuid_known = false;
  std::uint64_t vram_bytes = 0;     // plus grand tas DEVICE_LOCAL
  bool cooperative_matrix = false;  // VK_KHR_cooperative_matrix ou VK_NV_cooperative_matrix
};

// Noms du chargeur Vulkan, essayés dans l'ordre.
std::vector<std::string> vulkan_loader_names(core::Os os);

// Appareils dans l'ordre de l'énumération. Refus LibraryUnavailable sans chargeur,
// VulkanCallFailed (qui cite l'appel et le VkResult) si une instance ne se crée pas ;
// une liste vide si le système n'a aucun appareil.
std::expected<std::vector<VulkanDevice>, Refusal> probe_vulkan(const LibraryLoader& loader, core::Os os);

core::GpuVendor vendor_of(std::uint32_t pci_vendor);

// Version du pilote lisible : driver_info s'il est renseigné, sinon décodée selon
// le constructeur (NVIDIA 10.8.8.6 bits, Intel sous Windows 18.14 bits, sinon Vulkan).
std::string driver_version_text(const VulkanDevice& device, core::Os os);

std::string uuid_text(const DeviceUuid& uuid);  // 32 chiffres hexadécimaux, tirets du format 8-4-4-4-12

}  // namespace ajp::app
