// Profil d'un appareil pour le moteur de réseaux : famille, pilote, unités matricielles et ce
// que les noyaux peuvent y faire (net::Target), plus la clé des caches (autotuner, artefacts).
#pragma once

#include <string>

#include "gpu/vulkan/backend/family.h"
#include "gpu/vulkan/backend/net_program.h"
#include "gpu/vulkan/compute/context.h"

namespace ajp::gpu::vk {

struct DeviceProfile {
  GpuFamily family = GpuFamily::Unknown;
  DriverKind driver = DriverKind::Other;
  // Produit float16 × float16 + float32 en 16 × 16 × 16 par sous-groupe, activé sur le VkDevice.
  bool matrix_hardware = false;
  bool matrix_allowed = false;  // famille et pilote dans la liste d'autorisation
  std::string matrix_shapes;    // tuiles annoncées par le pilote (« 16×16×16 f16→f32, … »), ou vide
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

}  // namespace ajp::gpu::vk
