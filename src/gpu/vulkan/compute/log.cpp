#include "gpu/vulkan/compute/log.h"

#include "core/log.h"

namespace ajp::gpu::vk {

std::shared_ptr<spdlog::logger> make_logger() {
  return log_channel("vk_compute");
}

}  // namespace ajp::gpu::vk
