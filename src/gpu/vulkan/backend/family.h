// Familles de GPU et liste d'autorisation des unités matricielles. La famille se déduit des
// identifiants PCI (constructeur, appareil), jamais du nom ; le pilote se lit dans
// VkPhysicalDeviceDriverProperties. Les unités matricielles ne servent que sur une famille
// et un pilote qualifiés : un pilote qui se trompe ne casse pas l'image.
#pragma once

#include <cstdint>
#include <optional>
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
  IntelXeLp,   // Tiger Lake à Raptor Lake (UHD 770, Iris Xe), DG1
  IntelXeLpg,  // Meteor Lake et Arrow Lake (Arc intégré des Core Ultra 100 et 200)
  IntelAlchemist,
  IntelXe2,
  Apple,  // puces M d'Apple sous MoltenVK
};

std::string_view name(GpuFamily family);

// cpu : VkPhysicalDeviceType vaut VK_PHYSICAL_DEVICE_TYPE_CPU. device_name : un appareil AMD hors
// des plages connues se reconnaît à la puce que RADV nomme (« (RADV NAVI48) », « (RADV GFX1150) »).
GpuFamily detect_family(std::uint32_t vendor_id, std::uint32_t device_id, bool cpu,
                        std::string_view device_name = {});

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

// Unités matricielles (P2.46) : décidées sur les tuiles que le pilote annonce
// (vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR, device_profile.cpp) et sur la sortie de chaque couche,
// vérifiée par l'autotuner ; une seule liste d'exclusion écarte les appareils et pilotes connus pour les
// annoncer à tort (AMD avant RDNA3, Intel sans XMX, processeur, NVIDIA hors pilote propriétaire...).
// matrix_units_excluded en donne la raison. Intel Arc et Xe2 ne sont pas exclus, mais leur tuile 8×16×16
// n'entre pas encore dans net_conv_coopmat.glsl, qui ne tuile qu'en 16×16×16.
std::optional<std::string_view> matrix_units_excluded(GpuFamily family, DriverKind driver);
bool matrix_units_allowed(GpuFamily family, DriverKind driver);

// Les tuiles FP8 (coopmat_fp8) : jamais sur une RTX 50 (Blackwell) tant qu'aucune mesure ne les justifie
// (intention du produit : pas de FP8 sans mesure) ; le pilote les annonce pourtant. Pas non plus sur une famille
// inconnue, qui peut être une RTX 50 d'un identifiant hors des tables (C45-05). À rouvrir avec une mesure.
bool matrix_f8_allowed(GpuFamily family);

// Permises sur une famille inconnue (puce hors des plages PCI) : proposées et vérifiées par l'autotuner,
// jamais en tête des candidats.
bool matrix_units_unlisted(GpuFamily family, DriverKind driver);

}  // namespace ajp::gpu::vk
