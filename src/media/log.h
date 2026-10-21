// Canal de journal du module media : la convention est celle de core/log.h.
// En-tête interne, inclus par les .cpp du module seulement.
#pragma once

#include <memory>

#include "core/log.h"

namespace ajp::media {

// Un canal neuf, qui partage les puits installés par app ; chaque objet garde le sien.
inline std::shared_ptr<spdlog::logger> make_logger() {
  return log_channel("media");
}

}  // namespace ajp::media
