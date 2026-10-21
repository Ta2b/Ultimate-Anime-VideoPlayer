#include "media/log.h"

#include <spdlog/spdlog.h>

namespace ajp::media {

std::shared_ptr<spdlog::logger> make_logger() {
  return spdlog::default_logger()->clone("media");
}

}  // namespace ajp::media
