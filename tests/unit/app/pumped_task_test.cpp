#include "app/play/pumped_task.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <thread>

#include <catch2/catch_test_macros.hpp>

using namespace ajp;
using namespace ajp::app::play;

TEST_CASE("run_pumped_for : une lecture qui ne répond plus rend la main au bout du délai (P2.15)",
          "[app][pumped]") {
  const auto start = std::chrono::steady_clock::now();
  const auto slow = run_pumped_for<int>("lent", std::chrono::milliseconds(200), false, [] {
    std::this_thread::sleep_for(std::chrono::seconds(2));  // un partage réseau muet
    return 1;
  });
  REQUIRE_FALSE(slow);
  CHECK(slow.error().code == RefusalCode::IoFailed);
  CHECK(std::chrono::steady_clock::now() - start < std::chrono::seconds(1));
  const auto quick = run_pumped_for<int>("rapide", std::chrono::seconds(5), false, [] { return 7; });
  REQUIRE(quick);
  CHECK(*quick == 7);
}

TEST_CASE("run_pumped_for : les sondes laissées à leur sort sont gardées, bornées, puis attendues (A30-05)",
          "[app][pumped]") {
  REQUIRE(settle_abandoned_threads(std::chrono::seconds(5)));  // celle du cas précédent a fini
  std::atomic<bool> release = false;
  const auto blocked = [&release] {
    while (!release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return 1;
  };
  for (std::size_t i = 0; i < kMaxAbandonedThreads; ++i)
    CHECK_FALSE(run_pumped_for<int>("muet", std::chrono::milliseconds(20), false, blocked));
  CHECK(abandoned_running() == kMaxAbandonedThreads);
  // Une de plus : refusée tout de suite, sans thread ni descripteur de plus (A33-07).
  const auto start = std::chrono::steady_clock::now();
  const auto refused = run_pumped_for<int>("muet", std::chrono::seconds(5), false, blocked);
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::IoFailed);
  CHECK(std::chrono::steady_clock::now() - start < std::chrono::seconds(1));
  CHECK_FALSE(settle_abandoned_threads(std::chrono::milliseconds(50)));  // la sortie ne les attend pas
  release = true;
  CHECK(settle_abandoned_threads(std::chrono::seconds(5)));  // rendus : joints
  CHECK(abandoned_running() == 0);
}
