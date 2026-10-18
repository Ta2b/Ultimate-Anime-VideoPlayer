#include "vendor_hdr/log.h"

#include <memory>

#include <spdlog/spdlog.h>

namespace ajp::vendor_hdr {

spdlog::logger& log() {
  // Initialisé une fois, sans registre global : le logger est sûr entre threads.
  static const std::shared_ptr<spdlog::logger> channel = spdlog::default_logger()->clone("vendor_hdr");
  return *channel;
}

}  // namespace ajp::vendor_hdr
