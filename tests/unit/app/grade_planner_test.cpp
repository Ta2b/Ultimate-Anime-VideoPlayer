// Le planificateur de fond du grade v2 (app/play/grade_planner.h, chantier 8.6).
#include "app/play/grade_planner.h"

#include <chrono>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using namespace ajp::app::play;
using ajp::media::Shot;

namespace {

constexpr int kSize = 9;  // des LUT minuscules : les tests cuisent vite

Shot shot(std::int64_t start_s, std::int64_t end_s, float apl, bool fade = false, float apl_end = 0.0F) {
  Shot s;
  s.start_us = start_s * 1'000'000;
  s.end_us = end_s * 1'000'000;
  s.fade = fade;
  s.first.apl_nits = apl;
  s.last.apl_nits = fade ? apl_end : apl;
  s.meta.apl_mean_nits = fade ? 0.5F * (apl + apl_end) : apl;
  s.meta.max_scl_nits = 1000.0F;
  for (float& part : s.meta.mean.lum_hist) part = 1.0F / static_cast<float>(s.meta.mean.lum_hist.size());
  return s;
}

ajp::render::DisplayContract contract() {
  ajp::render::DisplayContract c{.reference_nits = 203.0, .panel_peak_nits = 1000.0};
  c.oled = true;
  c.abl = ajp::render::default_abl(c.panel_peak_nits, c.oled);
  return c;
}

// Cuit tout ce que le plan demande autour de pts (ce que fait le fil).
void bake_all(GradePlan& plan, std::int64_t pts_us) {
  for (const BakeJob& job : plan.wanted(pts_us))
    plan.store(job.key, GradePlan::bake(job, plan.look(), plan.lut_size()));
}

}  // namespace

TEST_CASE("planificateur : le plan courant et les deux suivants, datés de leur début, dans l'ordre",
          "[app]") {
  GradePlan plan(PlannerConfig{.lut_size = kSize});
  plan.set_contract(contract());
  plan.set_shots({shot(0, 4, 20.0F), shot(4, 9, 80.0F), shot(9, 12, 10.0F), shot(12, 20, 40.0F)});
  CHECK(plan.take_orders(1'000'000, 0).empty());  // rien de cuit
  bake_all(plan, 1'000'000);
  const auto orders = plan.take_orders(1'000'000, 0);
  REQUIRE(orders.size() == 3);
  CHECK(orders[0].from_pts_us == 0);
  CHECK(orders[1].from_pts_us == 4'000'000);
  CHECK(orders[2].from_pts_us == 9'000'000);
  CHECK(orders[1].until_pts_us == 9'000'000);
  CHECK(orders[0].lut->size() == static_cast<std::size_t>(kSize * kSize * kSize * 3));
  CHECK(*orders[0].lut != *orders[1].lut);        // un plan clair et un sombre n'ont pas la même courbe
  CHECK(plan.take_orders(1'000'000, 0).empty());  // déjà donnés
  // Le plan suivant arrive : seul le quatrième manque, puis il est donné.
  CHECK(plan.take_orders(5'000'000, 0).empty());
  bake_all(plan, 5'000'000);
  const auto next = plan.take_orders(5'000'000, 0);
  REQUIRE(next.size() == 1);
  CHECK(next[0].from_pts_us == 12'000'000);
}

TEST_CASE("planificateur : un fondu reçoit les LUT de son début et de sa fin", "[app]") {
  GradePlan plan(PlannerConfig{.lut_size = kSize});
  plan.set_contract(contract());
  plan.set_shots({shot(0, 3, 5.0F, true, 120.0F)});
  CHECK(plan.wanted(0).size() == 2);
  bake_all(plan, 0);
  const auto orders = plan.take_orders(0, 0);
  REQUIRE(orders.size() == 1);
  REQUIRE(orders[0].fade_to);
  CHECK(*orders[0].fade_to != *orders[0].lut);
}

TEST_CASE("planificateur : huit LUT gardées au plus, les plus récentes", "[app]") {
  GradePlan plan(PlannerConfig{.lut_size = kSize});
  plan.set_contract(contract());
  std::vector<Shot> shots;
  for (int i = 0; i < 12; ++i) shots.push_back(shot(i, i + 1, 10.0F + 10.0F * static_cast<float>(i)));
  plan.set_shots(shots);
  for (int i = 0; i < 12; ++i) {
    bake_all(plan, i * 1'000'000LL);
    (void)plan.take_orders(i * 1'000'000LL, 0);
    CHECK(plan.cached() <= kPlanCache);
  }
  // Les derniers plans restent cuits : y revenir ne demande rien.
  CHECK(plan.wanted(11'000'000).empty());
  CHECK(plan.wanted(10'000'000).empty());
  // Le début est sorti du cache.
  CHECK_FALSE(plan.wanted(0).empty());
}

TEST_CASE("planificateur : en direct, E avance de 0,25 diaphragme par seconde au plus", "[app]") {
  GradePlan plan(PlannerConfig{.lut_size = kSize});
  plan.set_contract(contract());
  plan.set_shots({shot(0, 100, 40.0F)});
  bake_all(plan, 0);
  REQUIRE(plan.take_orders(0, 0).size() == 1);
  plan.set_user_e(0.25);
  (void)plan.take_orders(0, 500'000);
  CHECK(plan.applied_e() == 0.125);
  // Le nouvel E demande une nouvelle cuisson, donnée dès qu'elle est prête.
  bake_all(plan, 0);
  const auto orders = plan.take_orders(0, 500'000);
  REQUIRE(orders.size() == 1);
  CHECK(orders[0].e == 0.125);
  (void)plan.take_orders(0, 2'000'000);
  CHECK(plan.applied_e() == 0.25);
  plan.set_user_e(-1.0);  // borné à −0,25
  (void)plan.take_orders(0, 10'000'000);
  CHECK(plan.applied_e() == -0.25);
}

TEST_CASE("planificateur : le fil cuit en fond, main reçoit les ordres", "[app]") {
  GradePlanner planner(PlannerConfig{.lut_size = kSize});
  planner.set_contract(contract());
  planner.set_shots({shot(0, 2, 20.0F), shot(2, 4, 60.0F)});
  std::vector<GradePlanOrder> orders;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (orders.size() < 2 && std::chrono::steady_clock::now() < deadline) {
    for (auto& order : planner.take_orders(0, 0)) orders.push_back(std::move(order));
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  REQUIRE(orders.size() == 2);
  CHECK(orders[1].from_pts_us == 2'000'000);
  CHECK(planner.baked() >= 2);
}
