// Bornes de la cadence (chantier D04, défauts B12-01 à B12-06) : chaque cas échouait avant sa correction.
#include <algorithm>
#include <cstdint>
#include <optional>

#include <catch2/catch_test_macros.hpp>

#include "core/auto_rate.h"
#include "core/free_grid.h"
#include "core/free_rate.h"
#include "core/pacing.h"
#include "core/steady_grid.h"
#include "core/steady_rate.h"

using namespace ajp;
using namespace ajp::core;

namespace {
constexpr double kFps = 24000.0 / 1001.0;
SourceStamp src(std::uint32_t id, std::int64_t pts) {
  return {.id = SourceId{id}, .pts_us = pts, .epoch = 0};
}
}  // namespace

TEST_CASE("B12-01 : en VRR, le Stable ne dépasse pas la fréquence de l'écran", "[core][cadence_bounds]") {
  for (const double hz : {120.0, 144.0, 165.0, 240.0}) {
    const AutoLimits limits = auto_limits(hz, kFps, /*vrr=*/true, /*level_factor=*/10);
    const double out_fps = kFps * limits.ceiling / limits.granularity;
    INFO("écran " << hz << " Hz : plafond " << limits.ceiling << "/" << limits.granularity << " = " << out_fps
                  << " i/s");
    CHECK(out_fps <= hz);
    CHECK(kFps * (limits.ceiling + 1) / limits.granularity > hz);  // et le palier d'après dépasse
  }
}

TEST_CASE("B12-02 : sur la fréquence mesurée, le facteur exact se cale", "[core][cadence_bounds]") {
  // 165 Hz annoncés, 164,78 mesurés : le facteur calculé sur l'annoncée ne se cale pas sur la mesure (écart
  // de 0,13 %, au-delà de kMaxSpeedChange). Le lecteur calcule donc les paliers sur la fréquence mesurée.
  const AutoLimits announced = auto_limits(165.0, kFps, /*vrr=*/false, /*level_factor=*/10);
  REQUIRE(announced.exact.size() >= 2);
  CHECK_FALSE(vsync_lock(164.78, kFps * announced.exact.back().value()));
  const AutoLimits measured = auto_limits(164.78, kFps, /*vrr=*/false, /*level_factor=*/10);
  REQUIRE(measured.exact.size() >= 2);
  for (const Ratio& factor : measured.exact) {
    if (factor == Ratio{1, 1}) continue;
    INFO("facteur " << factor.num << "/" << factor.den);
    CHECK(vsync_lock(164.78, kFps * factor.value()).has_value());
  }
}

TEST_CASE("B12-03 : le pas constant ne propose rien que la grille refuse", "[core][cadence_bounds]") {
  SteadyRate rate;
  rate.configure(FreeParams{}, FreeDisplay{.vrr = true, .hz = 360.0}, kFps);
  rate.prior(CostLine{.a = 500.0, .b = 1000.0});  // GPU rapide : tout tient dans le budget
  INFO("facteur choisi " << rate.factor().num << "/" << rate.factor().den);
  CHECK(rate.factor().value() > 1.0);
  CHECK(rate.factor().value() <= static_cast<double>(kMaxInstantsPerPair));
  SteadyGrid grid;
  std::size_t images = 0;
  for (std::uint32_t i = 0; i < 20; ++i)
    images += grid.push(src(i + 1, static_cast<std::int64_t>(i) * 41'708), rate.factor()).instants().size();
  CHECK(images > 150);  // 19 paires à 10 images ; 20 (la source seule) avant la correction
}

TEST_CASE("B12-04 : sur écran fixe, une source juste avant un vsync est montrée telle quelle",
          "[core][cadence_bounds]") {
  // 60 Hz, vsyncs en 0, 16 667, 33 334, 50 001… Source droite à 48 001 : 2 ms avant le vsync de 50 001.
  constexpr std::int64_t kVsync = 16'667;
  FreeGrid grid;
  const FreeInputs in{
      .min_gap_us = kVsync, .step_us = 8'000, .vsync_period_us = kVsync, .vsync_origin_us = 0};
  (void)grid.push(src(1, 6'293), in);
  const CadenceStep a = grid.push(src(2, 48'001), in);
  const CadenceStep b = grid.push(src(3, 89'709), in);
  bool source_shown = false;
  for (const OutputInstant& i : a.instants()) source_shown |= i.is_source();
  CHECK(source_shown);
  // La paire suivante ne recalcule pas la source 2 à t presque nul.
  const OutputInstant first = b.instants().front();
  INFO("premier instant de la paire suivante : " << first.t.num << "/" << first.t.den);
  CHECK((first.is_source() || first.t.value() > 0.25));
}

TEST_CASE("B12-05 : la cadence affichée du Rapide ne sous-compte pas la tranche en cours",
          "[core][cadence_bounds]") {
  FreeRate rate;
  rate.reset(0);
  for (std::int64_t t = 0; t <= 10'000'000; t += 10'000) rate.produced(1, t);  // 100 i/s pendant 10 s
  for (const std::int64_t at : {10'010'001LL, 10'125'000LL, 10'249'999LL}) {
    const FreeInfo info = rate.info(at);
    INFO("à " << at << " µs : " << info.fps << " i/s");
    CHECK(info.fps > 95.0);
    CHECK(info.fps < 105.0);
  }
}

TEST_CASE("B12-06 : le Stable suit un écran de 240 Hz jusqu'au bout", "[core][cadence_bounds]") {
  const AutoLimits fixed = auto_limits(240.0, kFps, /*vrr=*/false, /*level_factor=*/0);
  REQUIRE_FALSE(fixed.exact.empty());
  CHECK(fixed.exact.back().value() > 9.9);  // ×10,01 : une image par vsync
  const AutoLimits vrr = auto_limits(240.0, kFps, /*vrr=*/true, /*level_factor=*/0);
  CHECK(kFps * vrr.ceiling / vrr.granularity > 230.0);
}
