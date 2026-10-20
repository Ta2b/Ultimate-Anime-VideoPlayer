// Familles de GPU et liste d'autorisation des unités matricielles. La famille se déduit des
// identifiants PCI (constructeur, appareil), jamais du nom ; le pilote se lit dans
// VkPhysicalDeviceDriverProperties. Les unités matricielles ne servent que sur une famille
// et un pilote qualifiés : un pilote qui se trompe ne casse pas l'image.
#pragma once

#include <cstdint>
#include <string_view>

namespace ajp::gpu::vk {

enum class GpuFamily : std::uint8_t {
  Unknown,
  Software,  // lavapipe, SwiftShader : processeur
  NvidiaTuring,
  NvidiaAmpere,
  NvidiaAda,
  NvidiaBlackwell,
  AmdBeforeRdna3,
  AmdRdna3,
  AmdRdna4,
  IntelAlchemist,
  IntelXe2,
};

std::string_view name(GpuFamily family);

// cpu : VkPhysicalDeviceType vaut VK_PHYSICAL_DEVICE_TYPE_CPU.
GpuFamily detect_family(std::uint32_t vendor_id, std::uint32_t device_id, bool cpu);

// Pilotes connus, depuis VkPhysicalDeviceDriverProperties::driverName.
enum class DriverKind : std::uint8_t {
  Other,
  NvidiaProprietary,
  Radv,
  AmdProprietary,
  IntelMesa,
  IntelWindows
};
DriverKind driver_kind(std::string_view driver_name);

// Unités matricielles autorisées : NVIDIA Turing à Blackwell sous le pilote propriétaire
// (qualifié sur une RTX 4070 Ti), AMD RDNA3 et RDNA4 sous RADV ou le pilote AMD, en wave32 et
// vérifiées couche par couche par l'autotuner. Intel Xe2 : à qualifier avant d'entrer ici.
bool matrix_units_allowed(GpuFamily family, DriverKind driver);

}  // namespace ajp::gpu::vk
