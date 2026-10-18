#include "audio/log.h"

#include <spdlog/spdlog.h>

namespace ajp::audio {

std::shared_ptr<spdlog::logger> make_logger() {
  return spdlog::default_logger()->clone("audio");
}

}  // namespace ajp::audio
