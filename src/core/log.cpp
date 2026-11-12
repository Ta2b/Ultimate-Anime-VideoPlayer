#include "core/log.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <format>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <spdlog/async.h>
#include <spdlog/sinks/dist_sink.h>
#include <spdlog/spdlog.h>

namespace ajp {
namespace {

// Le répartiteur de install_log_sinks : un type à lui pour le reconnaître dans le journal
// par défaut, et ne jamais en empiler un second. En asynchrone, il dit aussi les lignes que la file pleine
// a jetées (B20-05) : sans cela, le journal perdait la ligne de cause d'une rafale sans que rien ne le
// montre.
class LogHub final : public spdlog::sinks::dist_sink_mt {
 public:
  // Posé avant que le journal asynchrone ne serve, jamais changé ensuite.
  // Ni fort ni faible : le pool tient les lignes, qui tiennent le logger, qui tient ce répartiteur ; un lien
  // fort empêcherait close_log de détruire le pool, donc de vider sa file. Un lien faible verrouillé ici
  // (weak_ptr::lock) donnait au fil du pool, le temps d'une ligne, la dernière référence pendant
  // spdlog::shutdown : le pool se détruisait alors sur son propre fil, qui ne peut pas se joindre, et
  // std::terminate arrêtait le processus à la fermeture (core_log_close, asan et gcc-13). Seul le fil du pool
  // appelle sink_it_, et le pool le joint avant de libérer sa file : le pointeur nu reste valide pour lui.
  void watch(spdlog::details::thread_pool* pool) { pool_ = pool; }

 protected:
  void sink_it_(const spdlog::details::log_msg& msg) override {
    if (pool_ != nullptr) {
      const std::size_t lost = pool_->overrun_counter();
      if (lost != seen_lost_) {
        const std::string text = std::format(
            "journal : {} ligne(s) perdue(s) avant celle-ci, la file était pleine",  // i18n-ok : journal
            lost - seen_lost_);
        seen_lost_ = lost;
        spdlog::sinks::dist_sink_mt::sink_it_(
            spdlog::details::log_msg(msg.logger_name, spdlog::level::warn, text));
      }
    }
    spdlog::sinks::dist_sink_mt::sink_it_(msg);
  }

 private:
  spdlog::details::thread_pool* pool_ = nullptr;  // à lui seul, jamais propriétaire (voir watch)
  std::size_t seen_lost_ = 0;                     // sous le verrou du répartiteur
};

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

std::shared_ptr<spdlog::logger> make_channel(std::string_view module, const spdlog::logger* source) {
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
  // Après close_log, plus de journal par défaut : un canal sans puits, qui ne dit rien et ne
  // plante pas (un thread détaché qui journalise après la fermeture, A18-01).
  if (source == nullptr) return std::make_shared<spdlog::logger>(std::string(module));
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
  std::shared_ptr<spdlog::logger> channel = make_channel(module, source.get());
  cache.channels.insert_or_assign(
      std::string(module),
      ChannelCache::Entry{.logger = channel, .generation = cache.generation, .source = source.get()});
  return channel;
}

void install_log_sinks(std::vector<spdlog::sink_ptr> sinks, spdlog::level::level_enum level,
                       bool asynchronous) {
  const std::shared_ptr<spdlog::logger> current = spdlog::default_logger();
  std::shared_ptr<LogHub> hub = hub_of(*current);
  if (hub == nullptr) {
    hub = std::make_shared<LogHub>();
    std::shared_ptr<spdlog::logger> routed;
    if (asynchronous) {
      spdlog::init_thread_pool(kLogQueue, 1);
      hub->watch(spdlog::thread_pool().get());
      routed = std::make_shared<spdlog::async_logger>(current->name(), hub, spdlog::thread_pool(),
                                                      spdlog::async_overflow_policy::overrun_oldest);
    } else {
      routed = std::make_shared<spdlog::logger>(current->name(), hub);
    }
    routed->set_level(level);
    // Une erreur part tout de suite dans le fichier : un arrêt brutal juste après ne la perd pas.
    routed->flush_on(std::min(current->flush_level(), spdlog::level::err));
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

namespace {

// Le texte de l'exception en cours, ou rien hors d'un gestionnaire.
std::string current_exception_text() {
  const std::exception_ptr caught = std::current_exception();
  if (caught == nullptr) return {};
  try {
    std::rethrow_exception(caught);
  } catch (const std::exception& error) {
    return error.what();
  } catch (...) {
    return "exception inconnue";
  }
}

}  // namespace

int run_logged(const std::function<int()>& body) {
  try {
    return body();
  } catch (...) {
    try {
      log_channel("app")->critical("arrêt sur une exception non rattrapée : {}", current_exception_text());
    } catch (...) {  // NOLINT(bugprone-empty-catch) : le journal lui-même a échoué, rien à dire de plus
    }
    return 1;
  }
}

void close_log() {
  // spdlog::shutdown ne vide aucun puits : le fichier reste tenu par les canaux du cache, son tampon
  // ne part qu'à un exit() normal, jamais avant abort() ni execv() (A18-01). Les puits sont retenus,
  // la file asynchrone vidée par shutdown (le pool écrit tout avant de s'arrêter), puis eux.
  std::vector<spdlog::sink_ptr> sinks;
  if (const std::shared_ptr<spdlog::logger> current = spdlog::default_logger()) sinks = current->sinks();
  {
    ModuleHub& routed = module_hub();
    const std::scoped_lock lock(routed.mutex);
    if (routed.hub != nullptr) sinks.push_back(routed.hub);
  }
  spdlog::shutdown();
  for (const spdlog::sink_ptr& sink : sinks) {
    try {
      sink->flush();
    } catch (...) {  // NOLINT(bugprone-empty-catch) : un puits qui ne se vide pas n'arrête pas la fermeture
    }
  }
  // Les canaux gardés lâchent les puits ; les suivants sont sans puits (make_channel).
  ChannelCache& cache = channel_cache();
  const std::scoped_lock lock(cache.mutex);
  cache.channels.clear();
  ++cache.generation;
}

void install_terminate_flush() {
  std::set_terminate([] {
    try {
      const std::string cause = current_exception_text();
      log_channel("app")->critical("std::terminate{}{}", cause.empty() ? "" : " : ", cause);
    } catch (...) {  // NOLINT(bugprone-empty-catch) : on s'arrête de toute façon
    }
    close_log();
    std::abort();
  });
}

}  // namespace ajp
