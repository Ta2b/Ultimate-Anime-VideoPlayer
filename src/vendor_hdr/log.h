// Canal de journal du module vendor_hdr : la convention est celle de core/log.h.
// En-tête interne, inclus par les .cpp du module seulement.
#pragma once

#include <memory>

#include "core/log.h"

namespace ajp::vendor_hdr {

// Un canal neuf, qui partage les puits installés par app ; chaque objet garde le sien.
inline std::shared_ptr<spdlog::logger> make_logger() {
  return log_channel("vendor_hdr");
}

}  // namespace ajp::vendor_hdr
