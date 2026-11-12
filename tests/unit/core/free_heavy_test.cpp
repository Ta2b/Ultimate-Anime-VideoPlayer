// Mode Rapide sur un écran de 165 Hz, scène lourde : l'écart que la grille planifie sur le temps du mouvement
// fort n'est pas une image manquée, et le repli ne vient que de pertes réelles.
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/free_grid.h"
#include "core/free_rate.h"

using namespace ajp::core;

namespace {

constexpr std::int64_t kSourcePeriod = 41'708;  // 23,976 i/s
constexpr std::int64_t kVsync = 6'061;          // 165 Hz

SourceStamp stamp(std::uint32_t id, std::int64_t pts) {
  return {.id = {id}, .pts_us = pts, .epoch = 0};
}

// Une paire calme coûte 4 ms par image, une paire en mouvement fort 9 ms.
FreeRate measured_rate(std::int64_t now) {
  FreeRate rate;
  rate.set_min_gap(kVsync);
  rate.cost(6, 6 * 4'000, now);
  rate.cost(4, 4 * 9'000, now, true);
  return rate;
}

// Les pts d'une scène lourde de `pairs` paires, planifiés par la grille libre sur le pas lourd.
std::vector<std::int64_t> heavy_scene(const FreeRate& rate, std::int64_t now, std::uint32_t pairs) {
  FreeGrid grid;
  grid.push(stamp(1, 0), {});
  const FreeInputs inputs{.min_gap_us = rate.min_gap_us(),
                          .step_us = rate.step_us(now, true),
                          .vsync_period_us = kVsync,
                          .vsync_origin_us = 0};
  std::vector<std::int64_t> pts{0};
  for (std::uint32_t k = 1; k <= pairs; ++k) {
    const CadenceStep step = grid.push(stamp(k + 1, k * kSourcePeriod), inputs);
    for (const OutputInstant& at : step.instants()) pts.push_back(*at.pts_us);
  }
  return pts;
}

}  // namespace

TEST_CASE("free 165 Hz : le pas lourd vaut deux vsyncs quand le pas calme en vaut un", "[free][vsync]") {
  const std::int64_t now = 10'000'000;
  const FreeRate rate = measured_rate(now);
  CHECK(rate.step_us(now) == kVsync);       // 4,4 ms : borné par l'écran
  CHECK(rate.step_us(now, true) == 9'900);  // 9 ms × 1,10
  CHECK(rate.info(now).step_us == kVsync);  // l'écart affiché (et lu par ChainStats) est le calme
  const std::vector<std::int64_t> pts = heavy_scene(rate, now, 120);
  // Sur la grille des vsyncs, une image sur deux vsyncs au moins : chaque écart dépasse d'un vsync la période
  // calme. Comptés contre elle, presque tous seraient des « manques ».
  std::int64_t against_calm = 0;
  for (std::size_t i = 1; i < pts.size(); ++i) {
    const std::int64_t gap = pts[i] - pts[i - 1];
    CHECK(gap >= kVsync);
    if ((gap + kVsync / 4) / kVsync > 1) ++against_calm;
  }
  CHECK(against_calm * 2 > static_cast<std::int64_t>(pts.size()));
}

TEST_CASE("free 165 Hz : sans perte réelle, la scène lourde ne replie pas et la pression reste à 1",
          "[free][vsync]") {
  const std::int64_t t0 = 10'000'000;
  FreeRate rate = measured_rate(t0);
  const std::vector<std::int64_t> pts = heavy_scene(rate, t0, 24 * 10);  // dix secondes
  // ~82 images par seconde produites, toutes montrées : aucune perte à donner.
  const auto per_bucket = static_cast<int>(pts.size() / 40);
  for (int i = 0; i < 40; ++i) {
    const std::int64_t now = t0 + i * FreeRate::kBucketUs;
    rate.produced(per_bucket, now);
    rate.missed(now, 0);
    CHECK_FALSE(rate.should_fall_back(now));
  }
  const std::int64_t end = t0 + 40 * FreeRate::kBucketUs;
  CHECK(rate.pressure(end) == 1.0);
  CHECK(rate.step_us(end, true) == 9'900);
  // L'ancien compte (un manque par écart lourd) : le repli tombait en quelques secondes.
  FreeRate old = measured_rate(t0);
  bool fell = false;
  for (int i = 0; i < 40 && !fell; ++i) {
    const std::int64_t now = t0 + i * FreeRate::kBucketUs;
    old.produced(per_bucket, now);
    old.missed(now, per_bucket);
    fell = old.should_fall_back(now);
  }
  CHECK(fell);
}

TEST_CASE("free 165 Hz : des pertes réelles et durables replient toujours", "[free][vsync]") {
  const std::int64_t t0 = 10'000'000;
  FreeRate rate = measured_rate(t0);
  // 20 images par tranche de 250 ms (80 i/s), une jetée en retard toutes les deux tranches : 2,5 %.
  for (int i = 0; i < 20; ++i) {
    rate.produced(20, t0 + i * FreeRate::kBucketUs);
    if (i % 2 == 0) rate.missed(t0 + i * FreeRate::kBucketUs, 1);
  }
  CHECK_FALSE(rate.should_fall_back(t0 + 19 * FreeRate::kBucketUs));
  // Puis deux par tranche (10 %) : le repli.
  for (int i = 20; i < 40; ++i) {
    rate.produced(20, t0 + i * FreeRate::kBucketUs);
    rate.missed(t0 + i * FreeRate::kBucketUs, 2);
  }
  CHECK(rate.should_fall_back(t0 + 39 * FreeRate::kBucketUs));
}
