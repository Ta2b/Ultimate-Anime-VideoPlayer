#include "gpu/ir/log.h"

#include <spdlog/spdlog.h>

namespace ajp::gpu::ir {

std::shared_ptr<spdlog::logger> make_logger() {
  return spdlog::default_logger()->clone("gpu_ir");
}

}  // namespace ajp::gpu::ir
