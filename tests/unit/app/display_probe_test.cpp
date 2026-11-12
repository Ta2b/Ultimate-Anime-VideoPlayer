// B11-05, B11-06, B32-02, B08-09 : les relevés de l'écran se font sur le fil de DisplayProbe ; le thread main
// demande et prend sans attendre.
#include "app/play/display_probe.h"

#include <atomic>
#include <chrono>
#include <optional>
#include <string>
#include <thread>

#include <catch2/catch_test_macros.hpp>

namespace ajp::app::play {
namespace {

using namespace std::chrono_literals;

struct Gate {
  std::atomic<bool> open{false};
  std::atomic<int> hz_reads{0};
  std::atomic<int> peak_reads{0};
  std::atomic<int> vrr_reads{0};
  std::atomic<int> wakes{0};
};

DisplayReaders slow_readers(Gate& gate) {
  return {.hz =
              [&gate](SDL_Window*, const DisplaySdl&) {
                while (!gate.open.load()) std::this_thread::sleep_for(1ms);
                ++gate.hz_reads;
                return 144.0;
              },
          .peak =
              [&gate](SDL_Window*, const DisplaySdl&) {
                ++gate.peak_reads;
                return std::optional<double>(1000.0);
              },
          .vrr =
              [&gate](SDL_Window*, const std::string& connector) {
                ++gate.vrr_reads;
                return std::optional<bool>(connector == "DP-1");
              }};
}

std::optional<DisplayReading> wait_reading(DisplayProbe& probe) {
  for (int i = 0; i < 2000; ++i) {
    if (auto reading = probe.take()) return reading;
    std::this_thread::sleep_for(1ms);
  }
  return std::nullopt;
}

void wake_count(void* context) {
  ++static_cast<Gate*>(context)->wakes;
}

}  // namespace

TEST_CASE("B11-05 : un relevé lent de l'écran ne retient pas le thread qui le demande", "[display_probe]") {
  Gate gate;
  DisplayProbe probe(nullptr, slow_readers(gate), &wake_count, &gate);
  const auto start = std::chrono::steady_clock::now();
  probe.ask({.hz = 60.0, .name = "x"}, true, false, std::string());
  CHECK_FALSE(probe.take().has_value());  // le relevé n'est pas fini : rien, sans attendre
  CHECK(std::chrono::steady_clock::now() - start < 50ms);
  CHECK(probe.busy());
  gate.open = true;
  const auto reading = wait_reading(probe);
  REQUIRE(reading);
  CHECK(reading->hz == 144.0);
  CHECK(reading->peak_read);
  CHECK(reading->peak_nits == 1000.0);
  CHECK_FALSE(reading->vrr_read);
  CHECK(gate.wakes.load() >= 1);          // la boucle est réveillée au relevé fini
  CHECK_FALSE(probe.take().has_value());  // pris une fois
}

TEST_CASE("B11-05 : les demandes faites pendant un relevé se fondent sans perdre le pic", "[display_probe]") {
  Gate gate;
  DisplayProbe probe(nullptr, slow_readers(gate), nullptr, nullptr);
  probe.ask({}, false, false, std::string());
  std::this_thread::sleep_for(5ms);           // le premier relevé tourne, retenu
  probe.ask({}, true, false, std::string());  // le pic demandé
  probe.ask({}, false, true, "DP-1");         // puis le VRR : le pic reste voulu
  gate.open = true;
  for (int i = 0; i < 2000 && probe.busy(); ++i) std::this_thread::sleep_for(1ms);
  REQUIRE_FALSE(probe.busy());
  const auto reading = probe.take();
  REQUIRE(reading);
  CHECK(reading->peak_read);
  CHECK(reading->vrr_read);
  CHECK(reading->vrr_armed == true);
  CHECK(gate.hz_reads.load() == 2);  // deux relevés, pas trois
  CHECK(gate.peak_reads.load() == 1);
}

}  // namespace ajp::app::play
