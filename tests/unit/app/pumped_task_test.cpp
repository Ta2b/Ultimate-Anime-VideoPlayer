#include "app/play/pumped_task.h"

#include <chrono>
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
