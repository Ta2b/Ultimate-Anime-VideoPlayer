#include "core/log.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <spdlog/sinks/callback_sink.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/spdlog.h>

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

TEST_CASE("Journal : un puits installé après coup reçoit les messages d'un canal déjà créé", "[log]") {
  Capture first;
  Capture later;
  ajp::install_log_sinks({first.sink()}, spdlog::level::info);
  const std::shared_ptr<spdlog::logger> media = ajp::log_channel("media");
  media->info("avant");

  ajp::install_log_sinks({later.sink()}, spdlog::level::info);
  media->info("après");
  ajp::log_channel("audio")->warn("nouveau canal");

  CHECK(first.lines() == std::vector<std::string>{"media : avant"});
  CHECK(later.lines() == std::vector<std::string>{"media : après", "audio : nouveau canal"});
}

TEST_CASE("Journal : le niveau installé filtre les canaux créés ensuite", "[log]") {
  Capture capture;
  ajp::install_log_sinks({capture.sink()}, spdlog::level::warn);
  const std::shared_ptr<spdlog::logger> subs = ajp::log_channel("subs");
  subs->info("ignoré");
  subs->error("gardé");
  CHECK(capture.lines() == std::vector<std::string>{"subs : gardé"});
  CHECK(subs->level() == spdlog::level::warn);
}

TEST_CASE("Journal : un canal n'entre pas au registre de spdlog", "[log]") {
  ajp::install_log_sinks({std::make_shared<spdlog::sinks::null_sink_mt>()}, spdlog::level::info);
  const std::shared_ptr<spdlog::logger> channel = ajp::log_channel("prepare");
  CHECK(channel->name() == "prepare");
  CHECK(spdlog::get("prepare") == nullptr);
  CHECK(channel != ajp::log_channel("prepare"));  // chaque appel rend son propre canal
}
