#include <cmath>
#include <cstdint>
#include <random>
#include <utility>

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

namespace {

// Des présentations à l'image neuve : 23,976 i/s sur un écran à `hz`, chacune au premier vblank après son
// échéance (tous les 6 ou 7 vblanks à 144 Hz), à ±jitter µs.
void feed_on_demand(VblankClock& clock, double hz, int frames, double jitter_us, std::int64_t from_us = 0) {
  const double period = 1e6 / hz;
  const double frame = 1e6 * 1001.0 / 24000.0;
  std::mt19937 rng(3);
  std::uniform_real_distribution<double> jitter(-jitter_us, jitter_us);
  for (int i = 0; i < frames; ++i) {
    const double due = frame * static_cast<double>(i);
    const double vblank = std::ceil(due / period) * period;
    clock.observe(from_us + 7'000'000 + static_cast<std::int64_t>(std::llround(vblank + jitter(rng))),
                  static_cast<std::uint64_t>(i), 0);
  }
}

}  // namespace

TEST_CASE("A12-01 : présentations à l'image neuve, la grille garde la période de l'écran",
          "[core][present]") {
  for (const double hz : {60.0, 120.0, 144.0, 164.78}) {
    VblankClock clock;
    clock.set_nominal_period(1e6 / std::round(hz));  // l'écran s'annonce à 60, 120, 144 ou 165 Hz
    feed_on_demand(clock, hz, 240, 150.0);           // dix secondes depuis zéro
    INFO("écran à " << hz << " Hz");
    CHECK(clock.hz() == Approx(hz).margin(0.01));
  }
}

TEST_CASE("A12-01 : sans période annoncée, un écart d'un vblank vu plus tard corrige la grille",
          "[core][present]") {
  VblankClock clock;
  feed_on_demand(clock, 144.0, 48, 0.0);
  // Rien ne dit l'écran : la grille ne peut que suivre les écarts vus (la cadence du contenu).
  CHECK(clock.hz() < 30.0);
  // L'interface s'ouvre et présente à chaque vblank : la grille repart sur le vrai pas.
  const double period = 1e6 / 144.0;
  const std::int64_t base = 7'000'000 + 3'000'000;
  for (int k = 0; k < 40; ++k)
    clock.observe(base + static_cast<std::int64_t>(std::llround(period * k)),
                  1000U + static_cast<unsigned>(k), 0);
  CHECK(clock.hz() == Approx(144.0).margin(0.05));
  // Et les présentations à la demande qui suivent ne la ramènent pas à la cadence du contenu.
  feed_on_demand(clock, 144.0, 120, 0.0, 4'000'000);
  CHECK(clock.hz() == Approx(144.0).margin(0.05));
}

TEST_CASE("A12-01 : une date hors de la grille est écartée, un autre mode fait repartir", "[core][present]") {
  const double period = 1e6 / 144.0;
  VblankClock clock;
  clock.set_nominal_period(period);
  for (int k = 0; k < 64; ++k)
    clock.observe(1'000'000 + static_cast<std::int64_t>(std::llround(period * 6 * k)),
                  static_cast<std::uint64_t>(k), 0);
  const std::size_t kept = clock.samples();
  // À mi-chemin entre deux vblanks (l'heure d'une soumission, pas d'un affichage) : écartée.
  clock.observe(1'000'000 + static_cast<std::int64_t>(std::llround(period * (6 * 63 + 2.5))), 99, 0);
  CHECK(clock.samples() == kept);
  CHECK(clock.hz() == Approx(144.0).margin(0.01));
  // L'écran passe à 60 Hz : les dates d'avant ne valent plus.
  clock.set_nominal_period(1e6 / 60.0);
  CHECK(clock.samples() == 0);
  CHECK_FALSE(clock.grid().has_value());
  // Une suite de dates qui ne tiennent pas la grille annoncée ne fait jamais de grille fausse.
  VblankClock wrong;
  wrong.set_nominal_period(1e6 / 60.0);
  for (int k = 0; k < 200; ++k)
    wrong.observe(1'000'000 + static_cast<std::int64_t>(std::llround(period * 1.37 * k)),
                  static_cast<std::uint64_t>(k), 0);
  if (const auto grid = wrong.grid()) CHECK(std::abs(grid->period_us * 60.0 / 1e6 - 1.0) <= 0.02);
}

namespace {

// La vitesse proposée (extrêmes après convergence) pour une dérive constante, des lectures toutes les
// interval_us dont une sur `lose_every` est perdue (0 : aucune).
std::pair<double, double> pll_speed_range(double drift, double interval_us, int lose_every) {
  ClockPll pll;
  double t = 0.0;
  double last = 0.0;
  double low = 2.0;
  double high = 0.0;
  for (int i = 0; i < 6000; ++i) {
    t += interval_us;
    if (lose_every > 0 && i % lose_every == 0) continue;  // lecture perdue : la suivante couvre deux pas
    const double speed = pll.update(drift * t, t - last).speed;
    last = t;
    if (i >= 4000) {
      low = std::min(low, speed);
      high = std::max(high, speed);
    }
  }
  return {low, high};
}

}  // namespace

TEST_CASE("A12-11 : la PLL audio est réglée en temps, pas en lectures", "[core][clock]") {
  const double drift = 100e-6;  // 100 ppm
  // 10 ms (WASAPI) et 21,3 ms (quantum PipeWire de 1 024 trames), avec et sans lecture perdue : la vitesse
  // proposée vaut la dérive à 1 ppm près.
  for (const double interval : {10'000.0, 21'333.0}) {
    for (const int lose_every : {0, 3}) {
      const auto [low, high] = pll_speed_range(drift, interval, lose_every);
      INFO("intervalle " << interval << " µs, une lecture perdue sur " << lose_every);
      CHECK(low == Approx(1.0 + drift).margin(1e-6));
      CHECK(high == Approx(1.0 + drift).margin(1e-6));
    }
  }
  // Même constante de temps quel que soit le périphérique : après un échelon de 5 ms, l'écart filtré a
  // parcouru la même part du chemin au bout d'une demi-seconde.
  const auto after_step = [](double interval) {
    ClockPll pll;
    pll.update(0.0, interval);
    double out = 0.0;
    for (double t = interval; t <= 500'000.0; t += interval) out = pll.update(5'000.0, interval).offset_us;
    return out;
  };
  CHECK(after_step(10'000.0) == Approx(after_step(21'333.0)).margin(250.0));
}
