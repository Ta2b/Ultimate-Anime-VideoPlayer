// Grille du Rapide à pas constant (core/steady_grid.h, chantier 7.5) : T1 (intervalles constants sur une
// heure, sans dérive), T6 (changements de N sans trou ni doublon), T7 (fréquence fixe : sur les vsyncs) et
// T9 relancé (dix minutes, GPU de 67 à 55 puis 120 i/s, N choisi par SteadyRate).
#include "core/steady_grid.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/steady_rate.h"

namespace ajp::core {
namespace {

// 23,976 i/s : le pts de la source k, arrondi au µs comme le démuxeur.
std::int64_t source_pts(std::int64_t k) {
  return (k * 1'001'000 + 12) / 24;
}

SourceStamp stamp(std::uint32_t id, std::int64_t pts, std::uint32_t epoch = 0) {
  return {.id = SourceId{id}, .pts_us = pts, .epoch = epoch};
}

struct Spread {
  std::int64_t min = INT64_MAX;
  std::int64_t max = INT64_MIN;
  double sum = 0.0;
  double sum2 = 0.0;
  long n = 0;
  void add(std::int64_t v) {
    min = std::min(min, v);
    max = std::max(max, v);
    sum += static_cast<double>(v);
    sum2 += static_cast<double>(v) * static_cast<double>(v);
    ++n;
  }
  double sigma() const {
    const double mean = sum / static_cast<double>(n);
    return std::sqrt(std::max(0.0, sum2 / static_cast<double>(n) - mean * mean));
  }
};

}  // namespace

TEST_CASE("steady grid : l'origine émet la source seule, puis N images par source", "[free][steady]") {
  SteadyGrid grid;
  const CadenceStep first = grid.push(stamp(0, 0), {5, 2});
  CHECK(first.event == GridEvent::Start);
  REQUIRE(first.instants().size() == 1);
  CHECK(first.instants()[0].is_source());
  std::size_t total = 0;
  for (std::uint32_t k = 1; k <= 4; ++k)
    total += grid.push(stamp(k, source_pts(k)), {5, 2}).instants().size();
  CHECK(total == 10);  // 4 paires × 2,5
}

TEST_CASE("steady grid : T1, une heure à N = 6,5, intervalles constants et sans dérive", "[free][steady]") {
  SteadyGrid grid;
  const Ratio n{13, 2};
  (void)grid.push(stamp(0, 0), n);
  std::int64_t last = 0;
  long emitted = 0;
  Spread spread;
  const std::int64_t sources = 24'000 * 3600 / 1001;  // une heure
  for (std::int64_t k = 1; k <= sources; ++k) {
    const CadenceStep step = grid.push(stamp(static_cast<std::uint32_t>(k), source_pts(k)), n);
    REQUIRE(step.event == GridEvent::Pair);
    for (const OutputInstant& at : step.instants()) {
      spread.add(*at.pts_us - last);
      last = *at.pts_us;
      ++emitted;
    }
  }
  INFO("intervalles " << spread.min << " à " << spread.max << " µs, σ " << spread.sigma() << " µs");
  CHECK(spread.max - spread.min <= 2);
  CHECK(spread.sigma() < 50.0);
  // L'instant j est à j·P/N exact : aucune dérive après une heure.
  const double exact = static_cast<double>(emitted) * 1'001'000.0 / 24.0 / 6.5;
  CHECK(std::abs(static_cast<double>(last) - exact) < 1.0);
}

TEST_CASE("steady grid : T6, N change sans trou, sans doublon, phase gardée", "[free][steady]") {
  SteadyGrid grid;
  const std::vector<Ratio> plan{{13, 2}, {4, 1}, {1, 1}, {3, 2}, {13, 2}, {1, 1}, {2, 1}};
  (void)grid.push(stamp(0, 0), plan[0]);
  std::int64_t last = 0;
  std::uint32_t id = 1;
  for (std::size_t p = 0; p < plan.size(); ++p) {
    const double step_us = 1'001'000.0 / 24.0 / plan[p].value();
    const double before_us = 1'001'000.0 / 24.0 / plan[p == 0 ? 0 : p - 1].value();
    for (int k = 0; k < 48; ++k, ++id) {
      const std::int64_t pts = source_pts(id);
      const CadenceStep step = grid.push(stamp(id, pts), plan[p]);
      REQUIRE(step.event == GridEvent::Pair);
      REQUIRE_FALSE(step.instants().empty());
      CHECK(step.instants().back().pts_us <= pts);
      for (const OutputInstant& at : step.instants()) {
        const std::int64_t gap = *at.pts_us - last;
        REQUIRE(gap > 0);
        // Au premier instant d'un nouveau N : entre l'ancien pas et le nouveau (à un 1/64 de période près).
        const double lo = std::min(step_us, before_us) - 700.0;
        const double hi = std::max(step_us, before_us) + 700.0;
        CHECK(static_cast<double>(gap) >= lo);
        CHECK(static_cast<double>(gap) <= hi);
        last = *at.pts_us;
      }
    }
    CHECK(grid.factor() == reduced(plan[p]));
  }
}

TEST_CASE("steady grid : à N = 1, la source seule ; une coupe repart de la source", "[free][steady]") {
  SteadyGrid grid;
  (void)grid.push(stamp(0, 0), {1, 1});
  const CadenceStep one = grid.push(stamp(1, source_pts(1)), {1, 1});
  CHECK(one.event == GridEvent::Pair);
  CHECK(one.left == SourceId{0});
  REQUIRE(one.instants().size() == 1);
  CHECK(one.instants()[0].is_source());
  grid.reset();
  CHECK(grid.push(stamp(2, source_pts(2)), {2, 1}).instants().size() == 1);  // origine
  CHECK(grid.push(stamp(3, source_pts(3)), {2, 1}).instants().size() == 2);
  // Époque nouvelle : origine ; époque passée : jetée.
  CHECK(grid.push(stamp(4, source_pts(4), 1), {2, 1}).event == GridEvent::NewEpoch);
  CHECK(grid.push(stamp(5, source_pts(5), 0), {2, 1}).event == GridEvent::Stale);
  CHECK_FALSE(grid.flush(0).has_value());
}

TEST_CASE("steady grid : T7, à 59,94 Hz les instants tombent sur les vsyncs", "[free][steady][vsync]") {
  SteadyRate rate;
  rate.configure({}, {.vrr = false, .hz = 60'000.0 / 1001.0}, 24'000.0 / 1001.0);
  rate.prior({.a = 500.0, .b = 500.0});
  const Ratio n = rate.factor();
  REQUIRE(n == Ratio{5, 2});
  SteadyGrid grid;
  (void)grid.push(stamp(0, 0), n);
  const double vsync = 1'001'000.0 / 60.0;
  double worst = 0.0;
  for (std::uint32_t k = 1; k < 24 * 600; ++k) {
    const CadenceStep step = grid.push(stamp(k, source_pts(k)), n);
    for (const OutputInstant& at : step.instants()) {
      const double r = std::fmod(static_cast<double>(*at.pts_us), vsync);
      worst = std::max(worst, std::min(r, vsync - r));
    }
  }
  CHECK(worst <= 2.0);
}

TEST_CASE("steady : T9 relancé, dix minutes, GPU de 67 à 55 puis 120 i/s", "[free][steady]") {
  // Le GPU sert les images une à une (coût d'une paire de n images : n × service) ; l'a priori est la phase
  // de départ ; une coupe toutes les 4 s. Le débit reste sous la FreeGrid (59,7 / 48,0 / 107,7 i/s), puisque
  // N est constant et pris avec 15 % de marge, mais sans perte et à intervalles constants.
  constexpr double kPeriod = 1'001'000.0 / 24.0;
  const auto service_at = [](double t_s) -> double {
    if (t_s < 200.0) return 14'925.0;
    if (t_s < 400.0) return 18'182.0;
    return 8'333.0;
  };
  SteadyRate rate;
  rate.configure({}, {.vrr = true, .hz = 165.0}, 24'000.0 / 1001.0);
  rate.prior({.a = 0.0, .b = 14'925.0});
  SteadyGrid grid;
  (void)grid.push(stamp(0, 0), rate.factor());
  std::vector<long> per_phase(3, 0);
  double backlog = 0.0;
  int lost = 0;
  long plan = 0;
  std::int64_t last = 0;
  const std::int64_t sources = 24'000 * 600 / 1001;
  for (std::int64_t k = 1; k <= sources; ++k) {
    const std::int64_t pts = source_pts(k);
    const double t_s = static_cast<double>(pts) / 1e6;
    if (static_cast<long>(t_s / 4.0) != plan) {
      plan = static_cast<long>(t_s / 4.0);
      rate.scene_cut(pts);
      grid.resync();
    }
    const Ratio n = rate.factor();
    (void)rate.take_changed();
    const CadenceStep step = grid.push(stamp(static_cast<std::uint32_t>(k), pts), n);
    const int images = static_cast<int>(step.instants().size());
    for (const OutputInstant& at : step.instants()) {
      REQUIRE(*at.pts_us > last);
      last = *at.pts_us;
    }
    const double gpu = service_at(t_s) * images;
    backlog = std::max(0.0, backlog + gpu - kPeriod);
    const double slot = kPeriod / n.value();
    if (backlog > 8 * slot) {
      const int l = static_cast<int>(std::ceil((backlog - 8 * slot) / slot));
      backlog = 8 * slot;
      lost += l;
      rate.lost(pts, l);
    }
    rate.pair_cost(images, static_cast<std::int64_t>(gpu), pts);
    rate.produced(images, pts);
    per_phase[static_cast<std::size_t>(std::min<std::int64_t>(pts / 200'000'000, 2))] += images;
  }
  const double fps0 = static_cast<double>(per_phase[0]) / 200.0;
  const double fps1 = static_cast<double>(per_phase[1]) / 200.0;
  const double fps2 = static_cast<double>(per_phase[2]) / 200.0;
  INFO("débits " << fps0 << " / " << fps1 << " / " << fps2 << " i/s, " << lost << " pertes");
  CHECK(fps0 >= 0.7 * 59.7);
  CHECK(fps1 >= 0.7 * 48.0);
  CHECK(fps2 >= 0.8 * 107.7);
  CHECK(lost <= 20);
  CHECK(sources - static_cast<std::int64_t>(last * 24 / 1'001'000) <= 1);  // pas de dérive
}

}  // namespace ajp::core
