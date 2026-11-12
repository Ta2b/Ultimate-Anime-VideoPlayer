// La file asynchrone du journal, dans un binaire à lui : le journal asynchrone ne se pose qu'une fois par
// processus (install_log_sinks), les autres tests du journal sont synchrones.
#include <atomic>
#include <chrono>
#include <cstddef>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <spdlog/sinks/callback_sink.h>

#include "core/log.h"

TEST_CASE("Journal : les lignes jetées par la file pleine sont comptées dans le journal (B20-05)", "[log]") {
  std::mutex mutex;
  std::vector<std::string> lines;
  std::atomic<bool> held{true};
  std::atomic<bool> first_seen{false};
  // Le disque qui ralentit : la première ligne bloque l'écriture tant que held tient.
  auto slow = std::make_shared<spdlog::sinks::callback_sink_mt>([&](const spdlog::details::log_msg& msg) {
    first_seen = true;
    while (held) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const std::scoped_lock lock(mutex);
    lines.emplace_back(msg.payload.begin(), msg.payload.end());
  });
  ajp::install_log_sinks({slow}, spdlog::level::info, true);
  ajp::log_channel("app")->info("première");
  while (!first_seen) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  constexpr std::size_t kExtra = 100;
  for (std::size_t i = 0; i < ajp::kLogQueue + kExtra; ++i) ajp::log_channel("app")->info("ligne {}", i);
  held = false;
  // La file se vide (la première, les kLogQueue gardées et la ligne du compte) avant la dernière, qui ne
  // déborde donc pas.
  for (int waited = 0; waited < 10000; ++waited) {
    {
      const std::scoped_lock lock(mutex);
      if (lines.size() >= ajp::kLogQueue + 2) break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ajp::log_channel("app")->info("dernière");
  ajp::close_log();
  const std::scoped_lock lock(mutex);
  bool counted = false;
  for (const std::string& line : lines)
    if (line.contains("perdue(s)"))
      counted = line.starts_with("journal : " + std::to_string(kExtra) + " ligne");
  CHECK(counted);
  CHECK(lines.back() == "dernière");
}
