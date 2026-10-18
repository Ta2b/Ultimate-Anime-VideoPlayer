#include "prepare/log.h"

#include <spdlog/spdlog.h>

namespace ajp::prepare {

std::shared_ptr<spdlog::logger> make_logger() {
  return spdlog::default_logger()->clone("prepare");
}

}  // namespace ajp::prepare
