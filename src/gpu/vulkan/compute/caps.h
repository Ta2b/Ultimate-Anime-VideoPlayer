// Capacités d'un appareil Vulkan pour le calcul : identité, fonctions activées, sous-groupes et
// limites. Le contexte les relève à la création ou à l'adoption, et les journalise.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "gpu/vulkan/compute/functions.h"
#include "gpu/vulkan/compute/vulkan.h"

namespace ajp::gpu::vk {

// Fonctions facultatives du calcul : disponibles sur l'appareil, ou activées sur le VkDevice.
struct Features {
  bool timeline_semaphore = false;    // obligatoire : soumission et attente par valeur
  bool storage_buffer_16bit = false;  // variantes « _f16 » des noyaux (tenseurs float16_t)
  bool shader_float16 = false;        // calcul en fp16 (réseaux)
  bool shader_int16 = false;
  bool subgroup_size_control = false;   // Vulkan 1.3
  bool compute_full_subgroups = false;  // Vulkan 1.3
  bool vulkan_memory_model = false;     // exigé par les shaders à matrices coopératives
  // Avec le modèle mémoire activé, toute portée Device (atomiques de block_diff, flow_max,
  // gmfss_splat) exige cette fonction : les deux vont ensemble, ou aucune des deux.
  bool vulkan_memory_model_device_scope = false;
  bool cooperative_matrix = false;  // VK_KHR_cooperative_matrix (unités matricielles)
  bool float_atomic_add = false;    // VK_EXT_shader_atomic_float : softmax splatting de GMFSS
};

struct DeviceCaps {
  std::string device_name;
  std::string driver_name;  // VkPhysicalDeviceDriverProperties
  std::string driver_info;
  std::uint32_t api_version = 0;  // version effective : la plus petite de l'instance et de l'appareil
  std::uint32_t driver_version = 0;
  std::uint32_t vendor_id = 0;
  std::uint32_t device_id = 0;
  VkPhysicalDeviceType type = VK_PHYSICAL_DEVICE_TYPE_OTHER;
  DeviceUuid uuid{};
  Features enabled;
  // Sous-groupes (Vulkan 1.1) : taille et opérations en calcul ; 0 si le calcul n'y a pas accès.
  std::uint32_t subgroup_size = 0;
  VkSubgroupFeatureFlags subgroup_operations = 0;
  // Tailles qu'un pipeline de calcul peut exiger (Vulkan 1.3, contrôle des sous-groupes : AMD 32 ou
  // 64, Intel 8 à 32) ; 0 si le calcul ne peut pas en exiger.
  std::uint32_t min_subgroup_size = 0;
  std::uint32_t max_subgroup_size = 0;
  // Limites utiles aux noyaux.
  std::uint32_t max_push_constants = 0;
  std::uint32_t max_workgroup_invocations = 0;
  std::array<std::uint32_t, 3> max_workgroup_count{};
  std::uint32_t max_shared_memory = 0;
  std::uint32_t max_storage_buffer_range = 0;
  VkDeviceSize min_storage_buffer_alignment = 0;
  std::uint32_t max_image_dimension_2d = 0;
  // Présentes sur l'appareil, sans être activées par le socle.
  bool cooperative_matrix = false;      // VK_KHR_cooperative_matrix
  std::vector<std::string> extensions;  // extensions activées sur le VkDevice
};

// Propriétés d'un appareil (rien n'est activé : enabled reste vide), pour une instance de
// version instance_api.
DeviceCaps probe_device(const InstanceFunctions& fn, VkPhysicalDevice device, std::uint32_t instance_api);

// Fonctions facultatives disponibles sur l'appareil.
Features available_features(const InstanceFunctions& fn, VkPhysicalDevice device, std::uint32_t api_version);

// Fonctions activées, lues dans la chaîne VkPhysicalDeviceFeatures2 passée à vkCreateDevice
// (celle de pl_vulkan.features par exemple) ; nullptr : rien de facultatif.
Features features_from_chain(const VkPhysicalDeviceFeatures2* chain);

// L'extension figure-t-elle dans la liste de l'appareil ?
bool has_extension(std::span<const VkExtensionProperties> extensions, const char* name);
std::vector<VkExtensionProperties> device_extensions(const InstanceFunctions& fn, VkPhysicalDevice device);

// « llvmpipe (Mesa 25.2.8), Vulkan 1.3, UUID 6d65…, timeline, stockage 16 bits… » : le
// rapport des capacités trouvées, une ligne.
std::string describe(const DeviceCaps& caps);

// UUID en hexadécimal, 32 chiffres.
std::string uuid_text(const DeviceUuid& uuid);

}  // namespace ajp::gpu::vk
