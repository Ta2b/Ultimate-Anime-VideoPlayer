#include "subs/log.h"

#include <spdlog/spdlog.h>

namespace ajp::subs {

std::shared_ptr<spdlog::logger> make_logger() {
  return spdlog::default_logger()->clone("subs");
}

}  // namespace ajp::subs
