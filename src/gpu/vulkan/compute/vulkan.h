// En-têtes Vulkan sans prototypes : aucune fonction n'est liée, chaque appel passe par les
// tables de functions.h, remplies depuis un vkGetInstanceProcAddr. La cible ajp::vk_compute
// définit VK_NO_PROTOTYPES pour tous ceux qui l'incluent.
#pragma once

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <array>
#include <cstdint>

#include <vulkan/vulkan.h>

namespace ajp::gpu::vk {

// UUID d'un appareil (VkPhysicalDeviceIDProperties::deviceUUID), le même type que
// gpu::DeviceUuid : le backend le compare à pl_gpu.uuid pour rendre son DeviceMatch.
using DeviceUuid = std::array<std::uint8_t, VK_UUID_SIZE>;

// Version minimale de l'instance et de l'appareil : timeline semaphores et stockage 16 bits
// sont du cœur en 1.2, comme pour libplacebo (PL_VK_MIN_VERSION). cmake/AjpSpirv.cmake et
// tools/ci/check-shaders.sh lisent cette ligne pour la cible des shaders (vulkan1.N) : en garder
// la forme « kMinApiVersion = VK_API_VERSION_1_N; ».
inline constexpr std::uint32_t kMinApiVersion = VK_API_VERSION_1_2;

}  // namespace ajp::gpu::vk
