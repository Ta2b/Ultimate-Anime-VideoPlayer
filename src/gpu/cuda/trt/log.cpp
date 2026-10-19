#include "gpu/cuda/trt/log.h"

#include <spdlog/spdlog.h>

namespace ajp::gpu::trt {

std::shared_ptr<spdlog::logger> make_logger() {
  return spdlog::default_logger()->clone("tensorrt");
}

}  // namespace ajp::gpu::trt
