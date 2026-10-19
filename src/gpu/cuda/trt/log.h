// Canal de journal du runtime TensorRT.
#pragma once

#include <memory>

#include <spdlog/logger.h>

namespace ajp::gpu::trt {

// Copie du journal par défaut nommée « tensorrt », hors du registre global de
// spdlog : chaque runtime garde la sienne.
std::shared_ptr<spdlog::logger> make_logger();

}  // namespace ajp::gpu::trt
