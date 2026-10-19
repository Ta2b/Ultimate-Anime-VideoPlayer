// Canal de journal du module gpu : la convention est celle de core/log.h.
// En-tête interne, inclus par les .cpp du module seulement.
#pragma once

#include <memory>

#include "core/log.h"

namespace ajp::gpu {

inline std::shared_ptr<spdlog::logger> make_logger() {
  return log_channel("gpu");
}

}  // namespace ajp::gpu
