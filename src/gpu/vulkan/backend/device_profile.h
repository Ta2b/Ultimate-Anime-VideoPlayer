// Profil d'un appareil pour le moteur de réseaux : famille, pilote, unités matricielles et ce
// que les noyaux peuvent y faire (net::Target), plus la clé des caches (autotuner, artefacts).
#pragma once

#include <span>
#include <string>

#include "gpu/vulkan/backend/family.h"
#include "gpu/vulkan/backend/net_program.h"
#include "gpu/vulkan/compute/context.h"

namespace kirari::gpu::vk {

struct DeviceProfile {
  GpuFamily family = GpuFamily::Unknown;
  DriverKind driver = DriverKind::Other;
  // Produit float16 × float16 + float32 en 16 × 16 × 16 par sous-groupe, activé sur le VkDevice.
  bool matrix_hardware = false;
  bool matrix_allowed = false;   // famille et pilote hors de la liste d'exclusion (matrix_units_excluded)
  bool matrix_unlisted = false;  // famille inconnue : proposées sans être en tête (matrix_units_unlisted)
  std::string matrix_shapes;     // tuiles annoncées par le pilote (« 16×16×16 f16→f32, … »), ou vide
  net::Target target;
  // « 10de:2782:NVIDIA:615.78.08:9a13a200 » : constructeur, appareil, nom, version lisible et
  // driverVersion du pilote. Les artefacts y sont liés : refaits à chaque pilote, ce qui coûte
  // peu tant que l'autotuner retrouve ses mesures.
  std::string key;
  // « 10de:2782:NVIDIA » : la même carte sous le même pilote, d'une version à l'autre. Les mesures
  // de l'autotuner y sont liées ; driver_build dit sous quelle version chacune a été vérifiée.
  std::string stable_key;
  std::string driver_build;  // « 615.78.08:9a13a200 »
};

DeviceProfile profile_device(const VkContext& context);

// Les tuiles de produits par sous-groupe qu'annonce le pilote
// (vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR) et dont un noyau se sert : 16 × 16 × 16 à sommes en
// float (net_conv_coopmat.glsl) ou en float16 (_acc16), FP8 et INT8 en 16 × 16 × 16, et 8 × 16 × 16 à sommes
// en float (_m8).
struct MatrixTiles {
  bool f32_sums = false;
  bool f16_sums = false;
  bool f8_e4m3 = false;  // FP8 E4M3 × E4M3 → float32 (VK_EXT_shader_float8) : RDNA 4, Blackwell
  bool s8 = false;       // INT8 × INT8 → int32 : RDNA 3 et 4, GeForce ; pas les Arc (M = 8)
  // float16 × float16 → float32 en 8 × 16 × 16 (M = 8) : Intel Xe2 sous ANV. Les Arc A (DG2) y annoncent
  // 8 × 8 × 16 (N = 8), qu'aucun noyau ne tuile.
  bool m8_f32_sums = false;
  std::string shapes;  // toutes celles du pilote, pour le journal
};
MatrixTiles matrix_tiles(std::span<const VkCooperativeMatrixPropertiesKHR> properties);

}  // namespace kirari::gpu::vk
