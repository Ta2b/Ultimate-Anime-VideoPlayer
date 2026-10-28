// Refus nommés du socle Vulkan : chaque VkResult en échec devient un Refusal qui cite l'appel.
#pragma once

#include <expected>
#include <string>
#include <string_view>
#include <utility>

#include "core/types.h"
#include "gpu/vulkan/compute/vulkan.h"

namespace ajp::gpu::vk {

// Résultat d'une opération qui ne rend rien d'autre qu'un succès ou un refus.
using Done = std::expected<void, Refusal>;

inline std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{.code = code, .detail = std::move(detail)});
}

// Nom de l'énumérateur, « VK_ERROR_DEVICE_LOST » ; « VkResult(n) » s'il est inconnu.
std::string result_name(VkResult result);

// VK_SUCCESS passe. Une mémoire épuisée (hôte ou appareil, pool de descripteurs) devient
// GpuMemoryExhausted ; tout autre résultat, VulkanCallFailed. Le détail : « appel : résultat ».
Done check(VkResult result, std::string_view call);

}  // namespace ajp::gpu::vk
