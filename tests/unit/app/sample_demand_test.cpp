// Le relevé qui ne tourne que tant qu'on le lit (sample_demand.h).
#include "app/play/sample_demand.h"

#include <atomic>
#include <chrono>
#include <stop_token>
#include <thread>

#include <catch2/catch_test_macros.hpp>

namespace ajp::app::play {
namespace {

using std::chrono::milliseconds;

// Attend qu'une condition devienne vraie, au plus deux secondes.
template <typename Condition>
bool eventually(Condition&& condition) {
  for (int i = 0; i < 400; ++i) {
    if (condition()) return true;
    std::this_thread::sleep_for(milliseconds(5));
  }
  return false;
}

}  // namespace

TEST_CASE("Relevé à la demande : il tourne tant qu'on le lit, dort ensuite, repart à la lecture (A24-22)",
          "[app][stats]") {
  SampleDemand demand;
  std::atomic<int> rounds = 0;
  std::atomic<bool> reading = true;
  std::jthread sampler([&](const std::stop_token& stop) {
    while (!stop.stop_requested()) {
      ++rounds;
      if (!demand.wait(stop, milliseconds(2), milliseconds(60))) break;
    }
  });
  // L'OSD affiché : il lit sans cesse, les tours s'enchaînent.
  std::jthread reader([&](const std::stop_token& stop) {
    while (!stop.stop_requested()) {
      if (reading) demand.asked();
      std::this_thread::sleep_for(milliseconds(5));
    }
  });
  REQUIRE(eventually([&] { return rounds >= 5; }));

  // L'OSD caché : plus de lecture, le relevé s'endort (au plus un tour encore en route).
  reading = false;
  std::this_thread::sleep_for(milliseconds(200));
  const int asleep = rounds;
  std::this_thread::sleep_for(milliseconds(300));
  CHECK(rounds <= asleep + 1);

  // Il reparaît : la première lecture réveille le relevé.
  const int before = rounds;
  demand.asked();
  CHECK(eventually([&] { return rounds > before; }));
}

TEST_CASE("Relevé à la demande : l'arrêt réveille un relevé endormi", "[app][stats]") {
  SampleDemand demand;
  std::atomic<bool> left = false;
  std::jthread sampler([&](const std::stop_token& stop) {
    while (demand.wait(stop, milliseconds(1), milliseconds(10))) {
    }
    left = true;
  });
  std::this_thread::sleep_for(milliseconds(50));  // jamais lu : il dort
  CHECK_FALSE(left);
  sampler.request_stop();
  CHECK(eventually([&] { return left.load(); }));
}

}  // namespace ajp::app::play
