// Le thread d'écriture (latest_writer.h) : par fichier, la dernière version demandée ; flush attend tout.
#include "app/play/latest_writer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace ajp::app::play {

TEST_CASE("Écritures : la dernière version de chaque fichier, faite avant la fin de flush", "[app][writer]") {
  std::mutex mutex;
  std::vector<std::string> written;
  std::atomic<bool> release{false};
  LatestWriter writer;
  // La première écriture tient le thread : les suivantes de même clé se remplacent pendant ce temps.
  writer.post("a", [&]() -> std::expected<void, Refusal> {
    while (!release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const std::scoped_lock lock(mutex);
    written.emplace_back("a0");
    return {};
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  for (int i = 1; i <= 3; ++i) {
    writer.post("a", [&, i]() -> std::expected<void, Refusal> {
      const std::scoped_lock lock(mutex);
      written.push_back("a" + std::to_string(i));
      return {};
    });
  }
  writer.post("b", [&]() -> std::expected<void, Refusal> {
    const std::scoped_lock lock(mutex);
    written.emplace_back("b");
    return std::unexpected(Refusal{RefusalCode::FileUnwritable, "refus pour le journal"});
  });
  release = true;
  writer.flush();
  const std::scoped_lock lock(mutex);
  REQUIRE(written.size() == 3);
  CHECK(written[0] == "a0");
  CHECK(std::ranges::find(written, "a3") != written.end());
  CHECK(std::ranges::find(written, "b") != written.end());
  CHECK(std::ranges::find(written, "a1") == written.end());
}

TEST_CASE("Écritures : un job qui lève ne bloque ni flush ni les suivantes", "[app][writer]") {
  LatestWriter writer;
  writer.post("a", []() -> std::expected<void, Refusal> { throw std::runtime_error("disque"); });
  writer.flush();
  std::atomic<bool> done{false};
  writer.post("b", [&]() -> std::expected<void, Refusal> {
    done = true;
    return {};
  });
  writer.flush();
  CHECK(done.load());
}

}  // namespace ajp::app::play
