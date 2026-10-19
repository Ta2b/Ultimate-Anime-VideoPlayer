// L'ordonnanceur face aux accidents : images en retard, seek, pause, pas image, ring recréé,
// entrées hors contrat.
#include <cstddef>
#include <cstdint>

#include <catch2/catch_test_macros.hpp>

#include "core/pacing.h"
#include "present/scheduler.h"
#include "scheduler_sim.h"

using ajp::ClockSnapshot;
using ajp::kNoPts;
using ajp::RefusalCode;
using ajp::RingFrame;
using ajp::core::CadenceChoice;
using ajp::core::DisplayInfo;
using ajp::core::Ratio;
using ajp::present::make_scheduler_config;
using ajp::present::PaceMode;
using ajp::present::PresentScheduler;
using ajp::present::SchedulerConfig;
using ajp::present::test::kNtscFilmFps;
using ajp::present::test::SimOptions;
using ajp::present::test::Simulation;

namespace {

CadenceChoice cadence(double hz, Ratio requested) {
  return ajp::core::choose_cadence({.refresh_hz = hz}, kNtscFilmFps, requested, 8.0).value();
}

// 23,976 im/s x5 sur 120 Hz : l'image j passe à la vsync j.
Simulation sim_120() {
  const auto choice = cadence(120.0, {5, 1});
  return Simulation(make_scheduler_config({.refresh_hz = 120.0}, choice, 7).value(),
                    SimOptions{.factor = choice.factor, .period_us = 1e6 / 120.0});
}

PresentScheduler grid_scheduler() {
  return PresentScheduler::create(
             {.mode = PaceMode::VsyncGrid, .vsyncs_per_frame = 1, .frame_period_us = 10'000.0})
      .value();
}

}  // namespace

TEST_CASE("gpu bloqué 3 vsyncs : l'image précédente se répète, puis la grille rattrape") {
  auto sim = sim_120();
  sim.set_delay([](std::int64_t frame) -> std::optional<std::int64_t> {
    if (frame == 100) return 103;
    return std::nullopt;
  });
  sim.run(200);
  const auto& stats = sim.scheduler().stats();
  CHECK(stats.repeated == 3);
  CHECK(stats.skipped >= 2);
  // La synchronisation ne glisse pas : passé l'accroc, l'image j revient à la vsync j.
  for (std::size_t n = 110; n < sim.records().size(); ++n)
    CHECK(sim.records()[n].frame == static_cast<std::int64_t>(n));
  for (std::size_t n = 100; n < 103; ++n) CHECK(sim.records()[n].frame == 99);
  CHECK(sim.releases_displayed() + sim.releases_unseen() + sim.busy_slots() == sim.produced());
}

TEST_CASE("image en retard d'une vsync à 144 Hz : montrée en retard, la suivante à l'heure") {
  const auto choice = cadence(144.0, {3, 1});
  Simulation sim(make_scheduler_config({.refresh_hz = 144.0}, choice, 7).value(),
                 SimOptions{.factor = choice.factor, .period_us = 1e6 / 144.0});
  // k = 2 : l'image j est due à la vsync 2j ; l'image 100 n'est prête qu'à la vsync 201.
  sim.set_delay([](std::int64_t frame) -> std::optional<std::int64_t> {
    if (frame == 100) return 201;
    return std::nullopt;
  });
  sim.run(260);
  const auto& stats = sim.scheduler().stats();
  CHECK(stats.repeated == 1);
  CHECK(stats.late == 1);
  CHECK(stats.skipped == 0);
  CHECK(sim.records()[200].frame == 99);
  CHECK(sim.records()[201].frame == 100);
  CHECK(sim.records()[202].frame == 101);
}

TEST_CASE("seek : les images de l'ancienne époque sont rendues sans être montrées") {
  auto sim = sim_120();
  sim.run(50);
  const auto before = sim.records().back().frame;
  sim.seek(1, 1000);
  sim.run(10);
  const auto& records = sim.records();
  // Jamais de noir : l'image d'avant le seek reste jusqu'à la première de l'époque 1, qui passe
  // dès la vsync suivante (un slot était libre : une montrée, deux en file, une rendue).
  for (const auto& record : records) CHECK(record.frame.has_value());
  CHECK(records[49].frame == before);
  REQUIRE(records[50].frame);
  CHECK(sim.epoch_of(*records[50].frame) == 1);
  CHECK(sim.pts_of(*records[50].frame) == ajp::present::test::source_pts(kNtscFilmFps, 1000));
  CHECK(sim.scheduler().stats().stale == 2);
  CHECK(sim.scheduler().stats().skipped == 0);
  // L'horloge repart de la source 1000 dans l'époque 1, et avance d'une image par vsync.
  const auto runs = sim.runs(50, records.size());
  for (std::size_t i = 0; i + 1 < runs.size(); ++i) CHECK(runs[i].second == 1);
  const ClockSnapshot& clock = records.back().clock;
  CHECK(clock.epoch == 1);
  CHECK(clock.media_us == sim.pts_of(*records.back().frame));
  CHECK(sim.releases_displayed() + sim.releases_unseen() + sim.busy_slots() == sim.produced());
}

TEST_CASE("pause : l'image reste, l'horloge s'arrête, rien n'est compté comme répété") {
  auto sim = sim_120();
  sim.run(100);
  sim.scheduler().set_paused(true);
  sim.run(50);
  const auto& records = sim.records();
  for (std::size_t n = 100; n < 150; ++n) {
    CHECK(records[n].frame == 99);
    CHECK(records[n].clock.rate == 0.0);
    CHECK(records[n].clock.media_us == records[99].clock.media_us);
  }
  sim.scheduler().set_paused(false);
  sim.run(50);
  // La lecture reprend où elle s'était arrêtée, une image par vsync.
  for (std::size_t n = 150; n < 200; ++n) CHECK(sim.records()[n].frame == static_cast<std::int64_t>(n - 50));
  CHECK(sim.scheduler().stats().repeated == 0);
  CHECK(sim.scheduler().stats().skipped == 0);
}

TEST_CASE("pas image en pause : une image de plus, et l'horloge prend son horodatage") {
  auto sim = sim_120();
  sim.run(20);
  sim.scheduler().set_paused(true);
  sim.run(5);
  sim.scheduler().step();
  sim.run(5);
  const auto& records = sim.records();
  CHECK(records[24].frame == 19);
  for (std::size_t n = 25; n < 30; ++n) CHECK(records[n].frame == 20);
  CHECK(records[29].clock.media_us == sim.pts_of(20));
  // Hors pause, step() ne fait rien.
  sim.scheduler().set_paused(false);
  sim.scheduler().step();
  sim.run(1);
  CHECK(sim.records().back().frame == 21);
}

TEST_CASE("une image sans horodatage passe à la vsync suivante") {
  auto scheduler = grid_scheduler();
  REQUIRE(scheduler.push({.slot = 0, .pts_us = 0, .ready_value = 1}));
  CHECK(scheduler.on_vsync({.index = 0}, {}).frame->slot == 0);
  REQUIRE(scheduler.push({.slot = 1, .pts_us = kNoPts, .ready_value = 1}));
  const auto decision = scheduler.on_vsync({.index = 1}, {});
  CHECK(decision.frame->slot == 1);
  REQUIRE(decision.releases.size() == 1);
  CHECK(decision.releases[0].displayed);
  CHECK(decision.releases[0].free.slot == 0);
  CHECK(decision.releases[0].free.semaphore_value == 2);
}

TEST_CASE("horodatage qui ne croît pas : refusé, et le slot rendu sans être montré") {
  auto scheduler = grid_scheduler();
  REQUIRE(scheduler.push({.slot = 0, .pts_us = 50'000, .ready_value = 3}));
  const auto refused = scheduler.push({.slot = 1, .pts_us = 50'000, .ready_value = 5});
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::SequenceOrder);
  const auto decision = scheduler.on_vsync({.index = 0}, {});
  REQUIRE(decision.releases.size() == 1);
  CHECK(decision.releases[0].free.slot == 1);
  CHECK(decision.releases[0].free.semaphore_value == 6);
  CHECK_FALSE(decision.releases[0].displayed);
  // Les slots remis ne reviennent pas à la décision suivante.
  CHECK(scheduler.on_vsync({.index = 1}, {}).releases.empty());
}

TEST_CASE("ring recréé : les images de l'ancien sont rendues, aucune n'est plus montrée") {
  auto scheduler = grid_scheduler();
  REQUIRE(scheduler.push({.slot = 0, .pts_us = 0, .ready_value = 1}));
  REQUIRE(scheduler.push({.slot = 1, .pts_us = 10'000, .ready_value = 1}));
  REQUIRE(scheduler.on_vsync({.index = 0}, {}).frame);
  scheduler.set_generation(1);
  // Une image encore en route de l'ancien ring : rendue, jamais montrée.
  REQUIRE(scheduler.push({.slot = 2, .generation = 0, .pts_us = 20'000, .ready_value = 1}));
  const auto decision = scheduler.on_vsync({.index = 1}, {});
  CHECK_FALSE(decision.frame);
  REQUIRE(decision.releases.size() == 3);
  CHECK(decision.releases[0].free.slot == 1);
  CHECK_FALSE(decision.releases[0].displayed);
  CHECK(decision.releases[1].free.slot == 0);
  CHECK(decision.releases[1].displayed);
  CHECK(decision.releases[2].free.slot == 2);
  CHECK_FALSE(decision.releases[2].displayed);
  for (const auto& release : decision.releases) CHECK(release.free.generation == 0);
  CHECK(scheduler.stats().stale == 2);
  REQUIRE(scheduler.push({.slot = 0, .generation = 1, .pts_us = 30'000, .ready_value = 1}));
  CHECK(scheduler.on_vsync({.index = 2}, {}).frame->generation == 1);
}

TEST_CASE("Flush : une époque qui recule est refusée, une époque future vaut Flush") {
  auto scheduler = grid_scheduler();
  REQUIRE(scheduler.flush(2));
  CHECK(scheduler.flush(1).error().code == RefusalCode::SequenceOrder);
  REQUIRE(scheduler.push({.slot = 0, .pts_us = 0, .epoch = 3, .ready_value = 1}));
  CHECK(scheduler.epoch() == 3);
  // Une image d'une époque passée part aussitôt, sans être montrée.
  REQUIRE(scheduler.push({.slot = 1, .pts_us = 0, .epoch = 2, .ready_value = 1}));
  const auto decision = scheduler.on_vsync({.index = 0}, {});
  CHECK(decision.frame->slot == 0);
  REQUIRE(decision.releases.size() == 1);
  CHECK(decision.releases[0].free.slot == 1);
  CHECK_FALSE(decision.releases[0].displayed);
}

TEST_CASE("file pleine : refusée, le slot rendu") {
  auto scheduler = grid_scheduler();
  for (std::uint32_t i = 0; i < ajp::present::FrameQueue::kCapacity; ++i) {
    REQUIRE(scheduler.push({.slot = i, .pts_us = 1000 * static_cast<std::int64_t>(i), .ready_value = 1}));
  }
  const auto refused = scheduler.push({.slot = 99, .pts_us = 1'000'000, .ready_value = 1});
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::FieldOutOfRange);
}

TEST_CASE("images tenues bornées : un Flush ne déborde jamais les slots à rendre") {
  using ajp::present::FrameQueue;
  using ajp::present::ReleaseList;
  auto scheduler = grid_scheduler();
  REQUIRE(scheduler.flush(1));
  for (std::uint32_t i = 0; i < FrameQueue::kCapacity; ++i) {
    REQUIRE(scheduler.push(
        {.slot = i, .pts_us = 10'000 * static_cast<std::int64_t>(i), .epoch = 1, .ready_value = 1}));
  }
  // Des images d'une époque passée, rendues aussitôt : elles restent tenues jusqu'à la décision.
  const std::size_t room = ReleaseList::kCapacity - FrameQueue::kCapacity;
  for (std::size_t i = 0; i < room; ++i) {
    REQUIRE(scheduler.push({.slot = static_cast<std::uint32_t>(100 + i), .epoch = 0, .ready_value = 1}));
  }
  // Une époque future jetterait toute la file dans une liste pleine : refusée avant tout Flush.
  const RingFrame next{.slot = 200, .pts_us = 0, .epoch = 2, .ready_value = 1};
  const auto refused = scheduler.push(next);
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::FieldOutOfRange);
  CHECK(scheduler.epoch() == 1);

  CHECK(scheduler.on_vsync({.index = 0}, {}).releases.size() == room);
  // Les slots remis restent valables jusqu'à la décision suivante, qui libère leur place.
  CHECK(scheduler.on_vsync({.index = 1}, {}).releases.size() == 1);
  REQUIRE(scheduler.push(next));
  CHECK(scheduler.epoch() == 2);
  // Les 30 images restées en file, puis celle montrée à la vsync 1, remplacée par la nouvelle.
  CHECK(scheduler.on_vsync({.index = 2}, {}).releases.size() == FrameQueue::kCapacity - 1);
}

TEST_CASE("configuration : bornes refusées, mode choisi par l'écran et le Plan") {
  const auto code = [](const SchedulerConfig& config) {
    return PresentScheduler::create(config).error().code;
  };
  CHECK(code({.mode = PaceMode::VsyncGrid, .vsyncs_per_frame = 0, .frame_period_us = 10'000.0}) ==
        RefusalCode::FieldOutOfRange);
  CHECK(code({.mode = PaceMode::VsyncGrid, .vsyncs_per_frame = 17, .frame_period_us = 10'000.0}) ==
        RefusalCode::FieldOutOfRange);
  CHECK(code({.mode = PaceMode::VsyncGrid, .vsyncs_per_frame = 1, .frame_period_us = 0.0}) ==
        RefusalCode::FieldOutOfRange);
  CHECK(code({.speed = 0.0}) == RefusalCode::FieldOutOfRange);

  const auto choice = cadence(120.0, {5, 1});
  const auto grid = make_scheduler_config({.refresh_hz = 120.0}, choice, 4).value();
  CHECK(grid.mode == PaceMode::VsyncGrid);
  CHECK(grid.generation == 4);
  CHECK(grid.frame_period_us == 1e6 / choice.output_fps);
  CHECK(make_scheduler_config({.refresh_hz = 120.0}, std::nullopt, 0)->mode == PaceMode::FixedTimestamps);
  CHECK(make_scheduler_config({.refresh_hz = 165.0, .vrr = true}, choice, 0)->mode == PaceMode::Vrr);
}
