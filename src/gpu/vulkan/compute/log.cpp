#include "gpu/vulkan/compute/log.h"

#include <spdlog/spdlog.h>

namespace ajp::gpu::vk {

std::shared_ptr<spdlog::logger> make_logger() {
  return spdlog::default_logger()->clone("vk_compute");
}

}  // namespace ajp::gpu::vk
