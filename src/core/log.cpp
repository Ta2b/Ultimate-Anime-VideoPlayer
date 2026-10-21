#include "core/log.h"

#include <string>
#include <utility>

#include <spdlog/sinks/dist_sink.h>
#include <spdlog/spdlog.h>

namespace ajp {
namespace {

// Le répartiteur de install_log_sinks : un type à lui pour le reconnaître dans le journal
// par défaut, et ne jamais en empiler un second.
class LogHub final : public spdlog::sinks::dist_sink_mt {};

std::shared_ptr<LogHub> hub_of(const spdlog::logger& logger) {
  if (logger.sinks().size() != 1) return nullptr;
  return std::dynamic_pointer_cast<LogHub>(logger.sinks().front());
}

}  // namespace

std::shared_ptr<spdlog::logger> log_channel(std::string_view module) {
  // Les puits du journal par défaut ne changent jamais en place (install_log_sinks remplace le
  // journal, ou le contenu de son répartiteur sous son verrou) et son niveau est atomique : le
  // copier pendant qu'un autre thread s'en sert est sûr.
  return spdlog::default_logger()->clone(std::string(module));
}

void install_log_sinks(std::vector<spdlog::sink_ptr> sinks, spdlog::level::level_enum level) {
  const std::shared_ptr<spdlog::logger> current = spdlog::default_logger();
  std::shared_ptr<LogHub> hub = hub_of(*current);
  if (hub == nullptr) {
    hub = std::make_shared<LogHub>();
    auto routed = std::make_shared<spdlog::logger>(current->name(), hub);
    routed->set_level(level);
    routed->flush_on(current->flush_level());
    spdlog::set_default_logger(std::move(routed));
  } else {
    current->set_level(level);
  }
  hub->set_level(level);
  hub->set_sinks(std::move(sinks));
}

}  // namespace ajp
