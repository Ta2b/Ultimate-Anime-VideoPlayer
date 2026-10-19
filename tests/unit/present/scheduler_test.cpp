// L'ordonnanceur sur les cadences du plan : 23,976 im/s vers 120 Hz (x5 et 5/3), 144 Hz (x3,
// 2 vsyncs par image), 165 Hz (11011/3200), 60 Hz (5/2), VRR, et la dérive de l'écran.
#include "present/scheduler.h"

#include <cmath>
#include <cstdint>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/pacing.h"
#include "scheduler_sim.h"

using ajp::ClockSnapshot;
using ajp::core::CadenceChoice;
using ajp::core::DisplayInfo;
using ajp::core::Grid;
using ajp::core::Ratio;
using ajp::present::make_scheduler_config;
using ajp::present::PaceMode;
using ajp::present::test::kNtscFilmFps;
using ajp::present::test::SimOptions;
using ajp::present::test::Simulation;

namespace {

DisplayInfo fixed(double hz) {
  return {.refresh_hz = hz};
}

DisplayInfo vrr(double hz) {
  return {.refresh_hz = hz, .vrr = true};
}

CadenceChoice cadence(const DisplayInfo& display, Ratio requested) {
  return ajp::core::choose_cadence(display, kNtscFilmFps, requested, 8.0).value();
}

// Vsync grid : chaque image tient exactement k vsyncs, dans l'ordre, sans rien sauter.
void check_vsync_grid(Simulation& sim, int k) {
  const auto runs = sim.runs(0, sim.records().size());
  REQUIRE(runs.size() > 10);
  for (std::size_t i = 0; i + 1 < runs.size(); ++i) {
    INFO("image " << runs[i].first);
    CHECK(runs[i].first == static_cast<std::int64_t>(i));
    CHECK(runs[i].second == k);
  }
  const auto& stats = sim.scheduler().stats();
  CHECK(stats.skipped == 0);
  CHECK(stats.repeated == 0);
  CHECK(stats.late == 0);
  CHECK(stats.stale == 0);
}

Simulation vsync_sim(const DisplayInfo& display, const CadenceChoice& choice, double real_hz) {
  const auto config = make_scheduler_config(display, choice, 7).value();
  REQUIRE(config.mode == PaceMode::VsyncGrid);
  return Simulation(config, SimOptions{.factor = choice.factor, .period_us = 1e6 / real_hz});
}

}  // namespace

TEST_CASE("23,976 im/s x5 sur 120 Hz : une image par vsync, horloge à la vitesse 1,001") {
  const auto choice = cadence(fixed(120.0), {5, 1});
  REQUIRE(choice.factor == Ratio{5, 1});
  REQUIRE(choice.grid == Grid::Vsync);
  REQUIRE(choice.vsyncs_per_frame == 1);
  auto sim = vsync_sim(fixed(120.0), choice, 120.0);
  sim.run(600);
  check_vsync_grid(sim, 1);

  // L'écran est maître : 8341,67 µs média par vsync de 8333,33 µs.
  const ClockSnapshot& clock = sim.records().back().clock;
  CHECK(clock.valid);
  CHECK(clock.rate == Catch::Approx(1.001).epsilon(1e-9));
  CHECK(clock.media_us == std::llround(599 * 1e6 / (kNtscFilmFps * 5.0)));
}

TEST_CASE("23,976 im/s 5/3 sur 120 Hz : 3 vsyncs par image") {
  const auto choice = cadence(fixed(120.0), {5, 3});
  REQUIRE(choice.factor == Ratio{5, 3});
  REQUIRE(choice.vsyncs_per_frame == 3);
  auto sim = vsync_sim(fixed(120.0), choice, 120.0);
  sim.run(900);
  check_vsync_grid(sim, 3);
}

TEST_CASE("23,976 im/s x3 sur 144 Hz : 2 vsyncs par image") {
  const auto choice = cadence(fixed(144.0), {3, 1});
  REQUIRE(choice.factor == Ratio{3, 1});
  REQUIRE(choice.vsyncs_per_frame == 2);
  auto sim = vsync_sim(fixed(144.0), choice, 144.0);
  sim.run(720);
  check_vsync_grid(sim, 2);
}

TEST_CASE("23,976 im/s sur 165 Hz : 11011/3200, t fractionnaire, 2 vsyncs par image") {
  const auto choice = cadence(fixed(165.0), {3, 1});
  REQUIRE(choice.factor == Ratio{11011, 3200});
  REQUIRE(choice.vsyncs_per_frame == 2);
  REQUIRE(choice.speed == Catch::Approx(1.0).epsilon(1e-12));
  auto sim = vsync_sim(fixed(165.0), choice, 165.0);
  sim.run(1650);
  check_vsync_grid(sim, 2);
  CHECK(sim.records().back().clock.rate == Catch::Approx(1.0).epsilon(1e-9));
}

TEST_CASE("23,976 im/s 5/2 sur 60 Hz : une image par vsync à 59,94 im/s") {
  const auto choice = cadence(fixed(60.0), {5, 2});
  REQUIRE(choice.factor == Ratio{5, 2});
  REQUIRE(choice.vsyncs_per_frame == 1);
  auto sim = vsync_sim(fixed(60.0), choice, 60.0);
  sim.run(600);
  check_vsync_grid(sim, 1);
}

TEST_CASE("dérive de l'écran : 164,9987 Hz réels pour 165 nominaux") {
  const auto choice = cadence(fixed(165.0), {3, 1});
  auto sim = vsync_sim(fixed(165.0), choice, 164.9987);
  sim.run(3300);
  // La grille compte des vsyncs, pas des microsecondes : rien ne glisse.
  check_vsync_grid(sim, 2);
  // L'horloge publiée avance au rythme réel de l'écran ; l'audio la suit.
  CHECK(sim.records().back().clock.rate == Catch::Approx(164.9987 / 165.0).epsilon(1e-9));
}

TEST_CASE("écran fixe sans calage : 23,976 im/s sur 60 Hz en 3:2") {
  const auto choice = ajp::core::fit_cadence(fixed(60.0), kNtscFilmFps, {1, 1}).value();
  REQUIRE(choice.grid == Grid::Timestamps);
  const auto config = make_scheduler_config(fixed(60.0), choice, 7).value();
  REQUIRE(config.mode == PaceMode::FixedTimestamps);
  Simulation sim(config, SimOptions{.period_us = 1e6 / 60.0});
  sim.run(600);
  const auto runs = sim.runs(0, sim.records().size());
  int twos = 0;
  int threes = 0;
  for (std::size_t i = 0; i + 1 < runs.size(); ++i) {
    CHECK(runs[i].first == static_cast<std::int64_t>(i));
    CHECK((runs[i].second == 2 || runs[i].second == 3));
    ++(runs[i].second == 2 ? twos : threes);
  }
  CHECK(std::abs(twos - threes) <= 1);
  CHECK(sim.scheduler().stats().skipped == 0);
  CHECK(sim.scheduler().stats().repeated == 0);
}

namespace {

// VRR : l'application interroge l'ordonnanceur toutes les millisecondes ; chaque image passe à
// la première interrogation où l'horloge maître a atteint son horodatage, une seule fois.
void check_vrr(Simulation& sim, double master_rate) {
  for (std::int64_t i = 0; i < 1000; ++i) sim.tick(i, i * 1000);
  std::int64_t expected = 0;
  for (const auto& record : sim.records()) {
    if (!record.changed) continue;
    REQUIRE(record.frame);
    CHECK(*record.frame == expected);
    const double media = static_cast<double>(record.host_us) * master_rate;
    const double pts = static_cast<double>(sim.pts_of(*record.frame));
    CHECK(media >= pts - 1.0);
    CHECK(media < pts + 1000.0 * master_rate + 1.0);
    ++expected;
  }
  CHECK(expected > 90);
  CHECK(sim.scheduler().stats().skipped == 0);
  CHECK(sim.scheduler().stats().late == 0);
}

}  // namespace

TEST_CASE("VRR x4 : chaque image à son horodatage sur l'horloge audio") {
  const auto choice = cadence(vrr(165.0), {4, 1});
  REQUIRE(choice.grid == Grid::Timestamps);
  const auto config = make_scheduler_config(vrr(165.0), choice, 7).value();
  REQUIRE(config.mode == PaceMode::Vrr);
  Simulation sim(config, SimOptions{.factor = choice.factor});
  sim.set_master(
      [](std::int64_t) { return ClockSnapshot{.media_us = 0, .host_us = 0, .rate = 1.0, .valid = true}; });
  check_vrr(sim, 1.0);
  // Vrr : present ne publie pas d'horloge, l'audio est maître.
  CHECK_FALSE(sim.records().back().clock.valid);
}

TEST_CASE("VRR : l'horloge audio dérive de 0,05 % sur l'hôte") {
  const auto choice = cadence(vrr(165.0), {4, 1});
  const auto config = make_scheduler_config(vrr(165.0), choice, 7).value();
  Simulation sim(config, SimOptions{.factor = choice.factor});
  sim.set_master([](std::int64_t) { return ClockSnapshot{.rate = 1.0005, .valid = true}; });
  check_vrr(sim, 1.0005);
}

TEST_CASE("VRR sans horloge audio valide : horloge hôte de repli") {
  const auto choice = cadence(vrr(165.0), {4, 1});
  const auto config = make_scheduler_config(vrr(165.0), choice, 7).value();
  Simulation sim(config, SimOptions{.factor = choice.factor});
  check_vrr(sim, 1.0);
}

TEST_CASE("VRR : prochaine échéance annoncée") {
  const auto choice = cadence(vrr(165.0), {4, 1});
  auto scheduler =
      ajp::present::PresentScheduler::create(make_scheduler_config(vrr(165.0), choice, 0).value()).value();
  const ClockSnapshot master{.media_us = 0, .host_us = 1'000'000, .rate = 1.0, .valid = true};
  REQUIRE(scheduler.push({.slot = 0, .pts_us = 0, .ready_value = 1}));
  REQUIRE(scheduler.push({.slot = 1, .pts_us = 10'427, .ready_value = 1}));
  const auto decision = scheduler.on_vsync({.host_us = 1'000'000}, master);
  REQUIRE(decision.frame);
  CHECK(decision.frame->slot == 0);
  REQUIRE(decision.next_due_host_us);
  CHECK(*decision.next_due_host_us == 1'010'427);
}
