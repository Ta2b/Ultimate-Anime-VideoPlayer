// Les voies de calcul en plus du backend Vulkan adopté (B10-04) : pur, pour placebo_open.cpp.
#pragma once

#include <algorithm>
#include <cstdint>

namespace ajp::app::play {

// libplacebo crée toutes les files de chaque famille qu'il emploie (VkDeviceQueueCreateInfo::queueCount =
// celui de la famille), mais pl_vulkan_queue::count ne compte que celles de son pool (queue_count, 1 ici).
// Les voies se comptent donc sur la famille : les files qui suivent celle du backend, au plus deux.
constexpr std::uint32_t backend_lanes(std::uint32_t family_queues, std::uint32_t backend_index) {
  return family_queues > backend_index + 1 ? std::min(family_queues - backend_index - 1, 2U) : 0;
}

}  // namespace ajp::app::play
