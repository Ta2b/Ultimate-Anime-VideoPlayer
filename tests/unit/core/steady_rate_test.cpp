// Rapide à pas constant par plan (core/steady_rate.h, chantier 7.4) : le choix de N, ses plafonds, et un banc
// de simulation (Scenario, run) qui rejoue des minutes de lecture : coût vrai des paires par plan, bruit,
// pointes, coût de l'affichage, anneau qui absorbe un retard puis perd des images.
#include "core/steady_rate.h"

#include <cmath>
#include <cstdint>
#include <random>

#include <catch2/catch_test_macros.hpp>

namespace ajp::core {
namespace {

constexpr double kFps = 24000.0 / 1001.0;

struct Scenario {
  double fps = kFps;
  FreeDisplay display{.vrr = true, .hz = 164.78};
  double a_us = 3000.0;  // coût vrai d'une paire du plan léger : a + b·n
  double b_us = 5000.0;
  double d_us = 0.0;  // affichage d'une image
  double cut_every_s = 4.0;
  int heavy_every = 0;  // un plan sur heavy_every est heavy_mul fois plus lourd ; 0 : aucun
  double heavy_mul = 1.8;
  bool random_plans = false;  // chaque plan tiré entre 0,7 et 1,6 fois le léger
  double noise = 0.03;        // écart relatif du coût d'une paire
  double spike_at_s = -1.0;
  double spike_len_s = 0.0;
  double spike_mul = 1.0;
  int ring = 8;
  double seconds = 300.0;
  double prior_mul = 1.0;  // l'a priori : le plan léger × prior_mul
  FreeParams params;
};

struct Outcome {
  int lost = 0;
  long produced = 0;
  double mean_n = 0.0;
  double optimum_n = 0.0;  // le meilleur candidat tenu sans marge, en moyenne
  std::uint32_t urgencies = 0;
};

Outcome run(const Scenario& s, std::uint32_t seed = 1) {
  SteadyRate rate;
  rate.configure(s.params, s.display, s.fps);
  rate.display_cost(static_cast<std::int64_t>(s.d_us));
  rate.prior(CostLine{.a = s.a_us * s.prior_mul, .b = s.b_us * s.prior_mul});
  std::mt19937 rng(seed);
  std::normal_distribution<double> gauss(0.0, 1.0);
  std::uniform_real_distribution<double> pick(0.7, 1.6);
  const double period = 1e6 / s.fps;
  const auto frames = static_cast<long>(s.seconds * s.fps);
  long plan = -1;
  double plan_mul = 1.0;
  double backlog = 0.0;
  double acc = 0.0;
  Outcome out;
  for (long k = 0; k < frames; ++k) {
    const double t_s = static_cast<double>(k) / s.fps;
    const auto now = static_cast<std::int64_t>(static_cast<double>(k) * period);
    const long this_plan = static_cast<long>(t_s / s.cut_every_s);
    if (this_plan != plan) {
      plan = this_plan;
      plan_mul = s.random_plans                                                   ? pick(rng)
                 : s.heavy_every > 0 && plan % s.heavy_every == s.heavy_every - 1 ? s.heavy_mul
                                                                                  : 1.0;
      if (k > 0) rate.scene_cut(now);
    }
    const double spike = t_s >= s.spike_at_s && t_s < s.spike_at_s + s.spike_len_s ? s.spike_mul : 1.0;
    const double n = rate.factor().value();
    const double before = acc;
    acc += n;
    const int images = static_cast<int>(std::floor(acc) - std::floor(before));
    const double gpu =
        (s.a_us + s.b_us * images) * plan_mul * spike * std::max(0.5, 1.0 + s.noise * gauss(rng));
    backlog = std::max(0.0, backlog + gpu + s.d_us * images - period);
    const double slot = period / n;
    const double slack = s.ring * slot;
    if (backlog > slack) {
      const int lost = static_cast<int>(std::ceil((backlog - slack) / slot));
      backlog = slack;
      out.lost += lost;
      rate.lost(now, lost);
    }
    rate.pair_cost(images, static_cast<std::int64_t>(gpu), now);
    rate.produced(images, now);
    out.produced += images;
    out.mean_n += n;
    // Le meilleur candidat que ce plan tient vraiment, sans marge.
    double best = 1.0;
    for (const Ratio& c : rate.candidates()) {
      if ((s.a_us + s.b_us * c.value()) * plan_mul * spike + s.d_us * c.value() <= period) best = c.value();
    }
    out.optimum_n += best;
  }
  out.mean_n /= static_cast<double>(frames);
  out.optimum_n /= static_cast<double>(frames);
  out.urgencies = rate.info().urgencies;
  return out;
}

}  // namespace

TEST_CASE("Rapide à pas constant : N ≤ 6,5 à 164,78 Hz", "[free][steady]") {
  SteadyRate rate;
  rate.configure({}, {.vrr = true, .hz = 164.78}, kFps);
  rate.prior({.a = 500.0, .b = 500.0});  // un GPU qui tiendrait bien plus
  CHECK(rate.factor() == Ratio{13, 2});
  CHECK(rate.candidates().back() == Ratio{13, 2});
  CHECK(rate.take_changed());
  CHECK_FALSE(rate.take_changed());
}

TEST_CASE("Rapide à pas constant : à fréquence fixe, les facteurs qui calent l'écran", "[free][steady]") {
  SteadyRate rate;
  rate.configure({}, {.vrr = false, .hz = 60.0}, kFps);
  // 60 / 23,976 = 2,5 ; 60 / (2 × 23,976) = 1,25 ; au-delà, plus rien au-dessus de 1.
  CHECK(rate.candidates() == std::vector<Ratio>{{1, 1}, {5, 4}, {5, 2}});
  rate.prior({.a = 500.0, .b = 500.0});
  CHECK(rate.factor() == Ratio{5, 2});
}

TEST_CASE("Rapide à pas constant : T10, d = 1,6 ms fait passer N de 6,5 à 5,0", "[free][steady]") {
  SteadyRate rate;
  rate.configure({}, {.vrr = true, .hz = 240.0}, kFps);
  rate.prior({.a = 3000.0, .b = 5000.0});
  CHECK(rate.factor() == Ratio{13, 2});
  rate.display_cost(1600);
  CHECK(rate.choose({.a = 3000.0, .b = 5000.0}) == Ratio{5, 1});
}

TEST_CASE("Rapide à pas constant : T4, coupes vers des plans 1,8 fois plus lourds", "[free][steady]") {
  Scenario s;
  s.heavy_every = 3;
  const Outcome o = run(s);
  INFO("pertes " << o.lost << " sur " << o.produced << ", N moyen " << o.mean_n << ", optimum " << o.optimum_n
                 << ", urgences " << o.urgencies);
  CHECK(static_cast<double>(o.lost) <= 0.002 * static_cast<double>(o.produced));
  CHECK(o.mean_n >= 0.75 * o.optimum_n);
}

TEST_CASE("Rapide à pas constant : T5, une pointe de 10 s, l'urgence borne les pertes", "[free][steady]") {
  Scenario s;
  s.cut_every_s = 1000.0;  // un seul plan : seule l'urgence peut réagir avant la coupe
  s.spike_at_s = 100.0;
  s.spike_len_s = 10.0;
  s.spike_mul = 1.6;
  s.seconds = 200.0;
  const Outcome with = run(s);
  Scenario without = s;
  without.params.urgent_losses = 1'000'000;
  const Outcome none = run(without);
  INFO("avec l'urgence " << with.lost << " pertes (" << with.urgencies << " urgences), sans " << none.lost);
  CHECK(with.lost <= 12);
  CHECK(static_cast<double>(with.lost) <= 0.2 * static_cast<double>(none.lost));
}

TEST_CASE("Rapide à pas constant : T8, plans tirés au hasard, N près de l'optimum", "[free][steady]") {
  Scenario s;
  s.random_plans = true;
  s.prior_mul = 1.3;  // un a priori pessimiste : la mesure doit le corriger
  const Outcome o = run(s, 7);
  INFO("pertes " << o.lost << " sur " << o.produced << ", N moyen " << o.mean_n << ", optimum "
                 << o.optimum_n);
  CHECK(o.mean_n >= 0.85 * o.optimum_n);
  CHECK(static_cast<double>(o.lost) <= 0.0005 * static_cast<double>(o.produced));
}

TEST_CASE("Rapide à pas constant : N reste constant dans un plan sans perte", "[free][steady]") {
  SteadyRate rate;
  rate.configure({}, {.vrr = true, .hz = 164.78}, kFps);
  rate.prior({.a = 3000.0, .b = 5000.0});
  (void)rate.take_changed();
  // Des paires un peu plus chères que l'a priori mais dans la marge : pas de changement en cours de plan.
  for (int i = 0; i < 23; ++i) rate.pair_cost(6 + i % 2, 3000 + 5000 * (6 + i % 2) + 300, i * 41'708);
  CHECK_FALSE(rate.take_changed());
}

TEST_CASE("Rapide à pas constant : le repli seulement à N = 1", "[free][steady]") {
  SteadyRate rate;
  rate.configure({}, {.vrr = true, .hz = 164.78}, kFps);
  rate.prior({.a = 3000.0, .b = 5000.0});
  std::int64_t now = 2'000'000;
  for (int i = 0; i < 100; ++i, now += 41'708) rate.produced(6, now);
  // Des pertes en rafale : l'urgence descend d'un demi-pas à chaque fois, jusqu'à 1.
  int guard = 0;
  while (!(rate.factor() == Ratio{1, 1}) && guard++ < 40) {
    CHECK_FALSE(rate.should_fall_back(now));
    rate.lost(now, 2);
    now += 10'000;
  }
  REQUIRE(rate.factor() == Ratio{1, 1});
  now += 5'000'000;  // hors du calme du démarrage, dont les pertes ne comptent pas (A12-08)
  for (int i = 0; i < 40; ++i, now += 41'708) {
    rate.produced(1, now);
    rate.lost(now, 1);
  }
  CHECK(rate.should_fall_back(now));
  CHECK(rate.info().frozen);
  rate.scene_cut(now);  // la coupe dégèle et repart de l'a priori
  CHECK_FALSE(rate.info().frozen);
}

TEST_CASE("Rapide à pas constant : sans a priori, N part de 1 et monte après quelques paires",
          "[free][steady]") {
  SteadyRate rate;
  rate.configure({}, {.vrr = true, .hz = 164.78}, kFps);
  CHECK(rate.factor() == Ratio{1, 1});
  // À N = 1, une paire de 1 image coûte a + b = 8 ms : la droite d'amorçage (0, 8 ms) tient 36,3 / 8 = 4,5.
  std::int64_t now = 0;
  for (int i = 0; i < 4; ++i, now += 41'708) rate.pair_cost(1, 8'000, now);
  CHECK(rate.factor() == Ratio{9, 2});
  // Le vrai coût à 4,5 (3 + 5 × 4,5 = 25,5 ms) reste dans le budget : aucune perte dans le banc.
  Scenario s;
  s.prior_mul = 0.0;  // aucun a priori
  const Outcome o = run(s);
  INFO("pertes " << o.lost << ", N moyen " << o.mean_n << ", optimum " << o.optimum_n);
  CHECK(static_cast<double>(o.lost) <= 0.002 * static_cast<double>(o.produced));
  CHECK(o.mean_n >= 0.75 * o.optimum_n);
}

TEST_CASE("A12-08 : les pertes du démarrage et des reprises ne replient pas le pas constant",
          "[free][steady]") {
  SteadyRate rate;
  rate.configure({}, {.vrr = true, .hz = 164.78}, kFps);  // sans a priori : N part de 1
  std::int64_t now = 2'000'000;
  // La première seconde perd une image par paire (moteurs qui chauffent) : aucun repli.
  for (int i = 0; i < 24; ++i, now += 41'708) {
    rate.produced(1, now);
    rate.lost(now, 1);
    CHECK_FALSE(rate.should_fall_back(now));
  }
  // Puis quatre secondes propres : toujours rien.
  for (int i = 0; i < 96; ++i, now += 41'708) rate.produced(1, now);
  CHECK_FALSE(rate.should_fall_back(now));
  // Un seek : les pertes de la reprise ne comptent pas non plus.
  rate.settle(now);
  for (int i = 0; i < 24; ++i, now += 41'708) {
    rate.produced(1, now);
    rate.lost(now, 1);
  }
  CHECK_FALSE(rate.should_fall_back(now));
  // En régime, un à-coup isolé (douze pertes dans la même tranche de 250 ms) ne replie pas ; des pertes
  // durables, si, comme pour le Rapide.
  now += 6'000'000;
  for (int i = 0; i < 100; ++i, now += 41'708) rate.produced(1, now);
  rate.lost(now, 12);
  CHECK_FALSE(rate.should_fall_back(now));
  for (int i = 0; i < 40; ++i, now += 41'708) {
    rate.produced(1, now);
    rate.lost(now, 1);
  }
  CHECK(rate.should_fall_back(now));
}

}  // namespace ajp::core
