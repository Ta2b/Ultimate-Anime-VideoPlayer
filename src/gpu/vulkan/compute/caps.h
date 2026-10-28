// Capacités d'un appareil Vulkan pour le calcul : identité, fonctions activées, sous-groupes et
// limites. Le contexte les relève à la création ou à l'adoption, et les journalise.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "gpu/vulkan/compute/functions.h"
#include "gpu/vulkan/compute/vulkan.h"

namespace ajp::gpu::vk {

// VK_EXT_shader_float8, absente des en-têtes Vulkan plus anciens : nom, type de structure et
// structure des fonctions (deux champs d'après la spécification ; des champs de réserve la
// protègent d'un pilote qui en écrirait davantage).
inline constexpr const char* kShaderFloat8ExtensionName = "VK_EXT_shader_float8";
inline constexpr auto kStructureTypeShaderFloat8Features = static_cast<VkStructureType>(1000567000);
struct ShaderFloat8Features {
  VkStructureType sType = kStructureTypeShaderFloat8Features;
  void* pNext = nullptr;
  VkBool32 shaderFloat8 = VK_FALSE;
  VkBool32 shaderFloat8CooperativeMatrix = VK_FALSE;
  VkBool32 reserved[6] = {};
};

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
  bool float8_matrix = false;       // VK_EXT_shader_float8 : matrices coopératives FP8 (RDNA 4)
};

// Adresse PCI d'un appareil (VK_EXT_pci_bus_info).
struct PciAddress {
  std::uint32_t domain = 0;
  std::uint32_t bus = 0;
  std::uint32_t device = 0;
  std::uint32_t function = 0;
  bool operator==(const PciAddress&) const = default;
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
  std::optional<PciAddress> pci;  // sans VK_EXT_pci_bus_info (lavapipe, MoltenVK) : absente
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

// « 0000:03:00.0 » : domaine, bus, appareil et fonction, comme lspci -D.
std::string pci_text(const PciAddress& pci);

// Un appareil tel qu'un autre runtime le voit (HIP, OpenVINO) : ce qu'il sait de son identité.
struct DeviceIdentity {
  std::optional<DeviceUuid> uuid;
  std::optional<PciAddress> pci;
};

// L'index, parmi others, de l'appareil Vulkan de caps : même UUID d'abord, sinon même adresse PCI
// (domaine, bus et appareil : HIP ne donne pas la fonction, toujours 0 pour un GPU) ; nullopt si
// aucun. Le repli par le bus sert là où les deux runtimes ne dérivent pas l'UUID de la même façon
// (RADV le tire du bus PCI, ROCm de l'identifiant unique du KFD) : sur un PC à plusieurs GPU AMD,
// l'UUID seul ne retrouvait aucun appareil et le premier était pris.
std::optional<std::size_t> match_device(std::span<const DeviceIdentity> others, const DeviceCaps& caps);

}  // namespace ajp::gpu::vk
