// install_module_log_sinks, dans un binaire à lui : le répartiteur qu'il pose vaut pour tout le
// binaire, et les tests du journal (log_test.cpp) supposent qu'il n'y en a pas.
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <spdlog/sinks/callback_sink.h>
#include <spdlog/spdlog.h>

#include "core/log.h"

namespace {

// Un puits qui garde « canal : message » de chaque ligne reçue.
class Capture {
 public:
  spdlog::sink_ptr sink() {
    return std::make_shared<spdlog::sinks::callback_sink_mt>([this](const spdlog::details::log_msg& msg) {
      const std::scoped_lock lock(mutex_);
      lines_.push_back(std::string(msg.logger_name.begin(), msg.logger_name.end()) + " : " +
                       std::string(msg.payload.begin(), msg.payload.end()));
    });
  }
  std::vector<std::string> lines() const {
    const std::scoped_lock lock(mutex_);
    return lines_;
  }

 private:
  mutable std::mutex mutex_;
  std::vector<std::string> lines_;
};

}  // namespace

// Le lecteur et un backend qui partagent spdlog (le spdlog.dll de vcpkg sous Windows) ont un seul
// journal par défaut : ici, un seul binaire, ce qui revient au même pour lui. Le rappel du lecteur
// écrit dans ce journal ; posé dans son répartiteur, il s'y appellerait lui-même sous son verrou.
TEST_CASE("Journal : les canaux d'un backend vont au rappel du lecteur sans toucher à son journal", "[log]") {
  Capture player;
  ajp::install_log_sinks({player.sink()}, spdlog::level::info);
  const std::vector<spdlog::sink_ptr> player_sinks = spdlog::default_logger()->sinks();

  // Le rappel BackendInit::log de app : log_channel("backend"), créé avant le backend.
  const std::shared_ptr<spdlog::logger> backend = ajp::log_channel("backend");
  std::vector<std::string> errors;
  backend->set_error_handler([&errors](const std::string& message) { errors.push_back(message); });
  auto relay =
      std::make_shared<spdlog::sinks::callback_sink_mt>([backend](const spdlog::details::log_msg& msg) {
        backend->info("{}", std::string_view(msg.payload.data(), msg.payload.size()));
      });
  ajp::install_module_log_sinks({relay}, spdlog::level::debug);
  CHECK(spdlog::default_logger()->sinks() == player_sinks);

  const std::shared_ptr<spdlog::logger> runtime = ajp::log_channel("onnxruntime");
  CHECK(runtime->level() == spdlog::level::debug);
  runtime->warn("nœuds laissés au CPU");
  backend->info("compilé");
  CHECK(errors.empty());
  CHECK(player.lines() == std::vector<std::string>{"backend : nœuds laissés au CPU", "backend : compilé"});

  // Un second backend : le répartiteur du binaire change de puits, ses canaux suivent.
  Capture later;
  ajp::install_module_log_sinks({later.sink()}, spdlog::level::debug);
  runtime->debug("rappel suivant");
  ajp::log_channel("vk_compute")->info("nouveau canal");
  CHECK(later.lines() ==
        std::vector<std::string>{"onnxruntime : rappel suivant", "vk_compute : nouveau canal"});
  CHECK(player.lines().size() == 2);
}
