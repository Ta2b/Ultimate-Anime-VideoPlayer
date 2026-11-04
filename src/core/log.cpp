#include "core/log.h"

#include <cstdint>
#include <map>
#include <mutex>
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

// Le répartiteur de install_module_log_sinks, à ce binaire seul, et le niveau de ses canaux.
struct ModuleHub {
  std::mutex mutex;
  std::shared_ptr<LogHub> hub;                            // sous mutex ; nul hors d'un backend
  spdlog::level::level_enum level = spdlog::level::info;  // sous mutex
};

ModuleHub& module_hub() {
  static ModuleHub module;
  return module;
}

// Un canal par module (1.11) : log_channel le rend sans recréer de logger à chaque appel. Une entrée vaut
// pour la génération des puits (install_log_sinks, install_module_log_sinks) et le journal par défaut d'où
// elle vient ; l'un ou l'autre changé, elle se refait.
struct ChannelCache {
  std::mutex mutex;
  std::uint64_t generation = 0;  // sous mutex
  struct Entry {
    std::shared_ptr<spdlog::logger> logger;
    std::uint64_t generation = 0;
    const spdlog::logger* source = nullptr;
  };
  std::map<std::string, Entry, std::less<>> channels;  // sous mutex
};

ChannelCache& channel_cache() {
  static ChannelCache cache;
  return cache;
}

void new_generation() {
  ChannelCache& cache = channel_cache();
  const std::scoped_lock lock(cache.mutex);
  ++cache.generation;
}

std::shared_ptr<spdlog::logger> make_channel(std::string_view module) {
  {
    ModuleHub& routed = module_hub();
    const std::scoped_lock lock(routed.mutex);
    if (routed.hub != nullptr) {
      auto channel = std::make_shared<spdlog::logger>(std::string(module), routed.hub);
      channel->set_level(routed.level);
      return channel;
    }
  }
  // Les puits du journal par défaut ne changent jamais en place (install_log_sinks remplace le
  // journal, ou le contenu de son répartiteur sous son verrou) et son niveau est atomique : le
  // copier pendant qu'un autre thread s'en sert est sûr.
  return spdlog::default_logger()->clone(std::string(module));
}

}  // namespace

std::shared_ptr<spdlog::logger> log_channel(std::string_view module) {
  const std::shared_ptr<spdlog::logger> source = spdlog::default_logger();
  ChannelCache& cache = channel_cache();
  const std::scoped_lock lock(cache.mutex);
  const auto found = cache.channels.find(module);
  if (found != cache.channels.end() && found->second.generation == cache.generation &&
      found->second.source == source.get())
    return found->second.logger;
  std::shared_ptr<spdlog::logger> channel = make_channel(module);
  cache.channels.insert_or_assign(
      std::string(module),
      ChannelCache::Entry{.logger = channel, .generation = cache.generation, .source = source.get()});
  return channel;
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
  new_generation();  // les canaux suivent le niveau
}

void install_module_log_sinks(std::vector<spdlog::sink_ptr> sinks, spdlog::level::level_enum level) {
  {
    ModuleHub& routed = module_hub();
    const std::scoped_lock lock(routed.mutex);
    if (routed.hub == nullptr) routed.hub = std::make_shared<LogHub>();
    routed.level = level;
    routed.hub->set_level(level);
    routed.hub->set_sinks(std::move(sinks));
  }
  new_generation();  // hors du verrou du répartiteur : log_channel prend le cache, puis lui
}

}  // namespace ajp
