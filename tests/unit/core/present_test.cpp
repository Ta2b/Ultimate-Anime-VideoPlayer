#include <cmath>
#include <cstdint>
#include <random>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/clock_pll.h"
#include "core/vblank_clock.h"
#include "core/vsync_index.h"
#include "core/wait_plan.h"

using Catch::Approx;
using namespace ajp::core;

TEST_CASE("vblank : 164,78 Hz retrouvés à 0,01 Hz sur 2 s à ±0,3 ms de gigue", "[core][present]") {
  const double period = 1e6 / 164.78;
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> jitter(-300.0, 300.0);
  VblankClock clock;
  std::int64_t k = 0;
  for (int i = 0; i < 330; ++i) {  // 2 s ; une image sur trois seulement est présentée
    k += 1 + (i % 3 == 0 ? 1 : 0);
    const auto visible =
        static_cast<std::int64_t>(5'000'000.0 + period * static_cast<double>(k) + jitter(rng));
    clock.observe(visible, static_cast<std::uint64_t>(i), visible - 12'000);
  }
  CHECK(clock.hz() == Approx(164.78).margin(0.01));
  CHECK(clock.output_latency_us() == 12'000);
  const auto grid = clock.grid();
  REQUIRE(grid.has_value());
  CHECK(grid->period_us == Approx(period).margin(0.1));
}

TEST_CASE("vblank : sans assez d'observations, pas de grille", "[core][present]") {
  VblankClock clock;
  for (int i = 0; i < 5; ++i) clock.observe(1'000'000 + i * 16'667, static_cast<std::uint64_t>(i), 0);
  CHECK_FALSE(clock.grid().has_value());
  CHECK(clock.hz() == 0.0);
  CHECK(clock.output_latency_us() == 0);
}

TEST_CASE("vsync : aucune alternance 1-2-1 près du milieu, gigue ±0,3 ms à 60 Hz", "[core][present]") {
  const double period = 16'666.67;
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> jitter(-300.0, 300.0);
  VsyncPicker picker;
  std::int64_t previous = 0;
  std::int64_t previous_gap = 0;
  int decreases = 0;
  int gaps_bad = 0;
  int repeats = 0;  // deux écarts égaux de suite : le motif 3-2 rompu
  // 24 im/s : une image toutes les 2,5 vblanks, une sur deux près du milieu. 10 min.
  for (int i = 0; i < 14'400; ++i) {
    const double x = 2.5 * i + jitter(rng) / period;
    const std::int64_t k = picker.pick(x);
    if (i > 0) {
      if (k < previous) ++decreases;
      const std::int64_t gap = k - previous;
      if (gap != 2 && gap != 3) ++gaps_bad;
      if (gap == previous_gap) ++repeats;
      previous_gap = gap;
    }
    previous = k;
  }
  CHECK(decreases == 0);
  CHECK(gaps_bad == 0);
  CHECK(repeats == 0);
  // Une dérive lente traverse le milieu : le sens d'arrondi change une fois, pas à chaque gigue.
  VsyncPicker drifting;
  int flips = 0;
  std::int64_t last_offset = -1;
  for (int i = 0; i < 2'000; ++i) {
    const double fraction = 0.3 + 0.4 * i / 2'000.0;  // de 0,3 à 0,7
    const std::int64_t offset = drifting.pick(100.0 + fraction + jitter(rng) / period) - 100;
    if (last_offset >= 0 && offset != last_offset) ++flips;
    last_offset = offset;
  }
  CHECK(flips == 1);
}

TEST_CASE("PLL : la gigue est filtrée, un saut est recalé, une dérive suivie", "[core][clock]") {
  std::mt19937 rng(3);
  std::uniform_real_distribution<double> jitter(-2000.0, 2000.0);
  ClockPll pll;
  double raw = 0.0;
  double filtered = 0.0;
  for (int i = 0; i < 2000; ++i) {
    const double m = jitter(rng);
    const auto out = pll.update(m, 20'000.0);
    if (i >= 200) {
      raw += m * m;
      filtered += out.offset_us * out.offset_us;
    }
  }
  CHECK(std::sqrt(filtered) < std::sqrt(raw) / 3.0);
  const auto jump = pll.update(50'000.0, 20'000.0);  // saut de 50 ms : recalé d'un coup
  CHECK(jump.offset_us == Approx(50'000.0));
  ClockPll drift;
  double error = 0.0;
  for (int i = 0; i < 400; ++i) {  // 0,1 % de dérive, lecture toutes les 20 ms
    const double m = 20.0 * i;
    error = std::abs(drift.update(m, 20'000.0).offset_us - m);
  }
  CHECK(error < 100.0);
  CHECK(drift.update(8000.0, 20'000.0).speed <= 1.0005);
}

TEST_CASE("attente : grossière jusqu'à 1,5 ms avant l'échéance, puis précise", "[core][present]") {
  CHECK(wait_plan(0, 10'000).coarse_ms == 8);
  CHECK(wait_plan(0, 10'000).precise_until_us == 10'000);
  CHECK(wait_plan(0, 1'000).coarse_ms == 0);
  CHECK(wait_plan(5'000, 1'000).coarse_ms == 0);
}
