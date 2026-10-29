#include "core/auto_rate.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using ajp::core::auto_ceiling;
using ajp::core::AutoRate;
using ajp::core::kFineGranularity;
using ajp::core::PairOutcome;

namespace {

constexpr std::int64_t kPeriod = 41'708;  // 23,976 i/s
constexpr std::int64_t kSecond = 1'000'000;

// Un GPU simulé : coût d'une paire de n images = a + b·n (µs) selon l'instant.
struct Gpu {
  double a = 8'000;
  double b = 4'000;
};

struct Trace {
  std::vector<std::pair<std::int64_t, int>> changes;  // (instant, nouveau palier)
};

// Fait tourner le contrôleur sur `seconds` de lecture ; la paire est en retard quand son coût dépasse la
// période, et chaque image manquante est signalée à l'écran. Rend les changements de palier.
Trace run(AutoRate& rate, std::int64_t& now, int seconds, const std::function<Gpu(std::int64_t)>& gpu_at,
          std::int64_t t0 = 0) {
  Trace trace;
  const std::int64_t end = now + seconds * kSecond;
  std::int64_t next_poll = now + 500'000;
  int last = rate.step();
  std::int64_t missed_since_poll = 0;
  std::uint32_t phase = 0;
  while (now < end) {
    now += kPeriod;
    const Gpu gpu = gpu_at(now - t0);
    const int step = rate.step();
    const int g = rate.granularity();
    // Images créées : le palier en moitiés ou au pas fin alterne entre deux nombres entiers.
    const int whole = step / g;
    const int frac = step % g;
    phase += static_cast<std::uint32_t>(frac);
    int images = whole;
    if (phase >= static_cast<std::uint32_t>(g)) {
      phase -= static_cast<std::uint32_t>(g);
      ++images;
    }
    images = std::max(images, 1);
    const double cost = images > 1 ? gpu.a + gpu.b * images : gpu.b;
    const bool late = cost > static_cast<double>(kPeriod);
    if (late && cost > 1.2 * static_cast<double>(kPeriod)) ++missed_since_poll;
    rate.pair(PairOutcome{.now_us = now, .made_step = step, .late = late ? 1 : 0, .period_us = kPeriod});
    if (images > 1) rate.cost(images, static_cast<std::int64_t>(cost), now);
    if (now >= next_poll) {
      next_poll += 500'000;
      rate.missed(now, missed_since_poll);
      missed_since_poll = 0;
      rate.tick(now, true);
    }
    if (rate.step() != last) {
      last = rate.step();
      trace.changes.emplace_back(now, last);
    }
  }
  return trace;
}

int changes_after(const Trace& t, std::int64_t from) {
  return static_cast<int>(std::ranges::count_if(t.changes, [&](const auto& c) { return c.first >= from; }));
}

}  // namespace

TEST_CASE("auto : le plafond suit l'écran", "[core][auto_rate]") {
  CHECK(auto_ceiling(165.0, 23.976) == 6);  // 143,86 i/s sous 165 Hz
  CHECK(auto_ceiling(120.0, 23.976) == 5);  // 119,88
  CHECK(auto_ceiling(60.0, 23.976) == 2);
  CHECK(auto_ceiling(144.0, 24.0) == 6);
  CHECK(auto_ceiling(0.0, 24.0) == 1);
  CHECK(auto_ceiling(165.0, 1000.0) == 1);  // cadence source absurde
  CHECK(auto_ceiling(165.0, 2.0) == 1);
}

TEST_CASE("auto : les demi-paliers et le pas fin suivent l'écran", "[core][auto_rate]") {
  CHECK(auto_ceiling(165.0, 23.976, 2) == 12);  // ×6 reste le maximum
  CHECK(auto_ceiling(120.0, 23.976, 2) == 10);  // ×5 : 119,88 tient, ×5,5 non
  CHECK(auto_ceiling(100.0, 23.976, 2) == 8);   // ×4,5 : 107,9 ne tient pas
  CHECK(auto_ceiling(110.0, 24.0, 2) == 9);
  CHECK(auto_ceiling(0.0, 24.0, 2) == 2);
  CHECK(auto_ceiling(100.0, 24.0, kFineGranularity) == 101);  // ×4,2 : 100,8 i/s, à 1 % près
  CHECK(auto_ceiling(0.0, 24.0, kFineGranularity) == kFineGranularity);
  CHECK(ajp::core::auto_granularity(7) == kFineGranularity);
  CHECK(ajp::core::auto_granularity(0) == 1);
}

TEST_CASE("auto : les images manquées font descendre vite", "[core][auto_rate]") {
  CHECK(ajp::core::auto_overloaded(707, 7900));
  CHECK_FALSE(ajp::core::auto_overloaded(29, 5731));
  AutoRate rate(6);
  std::int64_t now = 10 * kSecond;
  rate.seed(400, now);  // ×4 mémorisé : départ ×3
  CHECK(rate.step() == 3);
  CHECK(rate.missed(now, 2) == 3);            // deux ne suffisent pas
  CHECK(rate.missed(now + 500'000, 1) == 2);  // trois en 1 s : un palier de moins, tout de suite
  CHECK(rate.reason() == ajp::core::AutoReason::Missed);
  // Les manquées du temps de repos (images du palier d'avant) ne comptent pas.
  CHECK(rate.missed(now + 600'000, 5) == 2);
}

TEST_CASE("auto : deux paires en retard en une seconde descendent, un retard isolé non",
          "[core][auto_rate]") {
  AutoRate rate(6);
  std::int64_t now = 5 * kSecond;
  const auto late_pair = [&](int step, int late) {
    now += kPeriod;
    return rate.pair({.now_us = now, .made_step = step, .late = late, .period_us = kPeriod});
  };
  CHECK(late_pair(2, 1) == 2);
  now += 1'500'000;
  CHECK(late_pair(2, 1) == 2);  // le premier est trop vieux
  CHECK(late_pair(2, 1) == 1);  // deux en 1 s
  CHECK(late_pair(2, 1) == 1);  // paire faite au palier d'avant : ignorée, aucune cascade
}

TEST_CASE("auto : une scène facile monte au maximum, une lourde descend, puis tout remonte",
          "[core][auto_rate]") {
  AutoRate rate(6);
  std::int64_t now = 0;
  const auto easy = Gpu{.a = 6'000, .b = 4'000};     // ×6 : 30 ms
  const auto heavy = Gpu{.a = 14'000, .b = 10'000};  // ×2 : 34 ms ; ×3 : 44 ms
  const auto gpu = [&](std::int64_t t) { return t >= 20 * kSecond && t < 50 * kSecond ? heavy : easy; };
  const Trace trace = run(rate, now, 80, gpu);
  // Facile : le maximum en quelques secondes.
  int step_at_20 = 0;
  int step_at_50 = 0;
  for (const auto& [at, step] : trace.changes) {
    if (at <= 20 * kSecond) step_at_20 = step;
    if (at <= 50 * kSecond) step_at_50 = step;
  }
  CHECK(step_at_20 == 6);
  CHECK(step_at_50 <= 2);   // lourde : ×2 (ou ×1 un instant)
  CHECK(rate.step() == 6);  // facile à nouveau : le maximum
  // En régime (20 dernières s de chaque phase), plus aucun changement.
  CHECK(changes_after(trace, 70 * kSecond) == 0);
  // Jamais plus d'un changement par seconde en montée.
  for (std::size_t i = 1; i < trace.changes.size(); ++i) {
    if (trace.changes[i].second > trace.changes[i - 1].second)
      CHECK(trace.changes[i].first - trace.changes[i - 1].first >= AutoRate::kMinRiseGapUs);
  }
  // Pas d'oscillation : peu de changements au total (3 phases).
  CHECK(trace.changes.size() <= 20);
}

TEST_CASE("auto : sous charge constante le palier se pose et ne bouge plus", "[core][auto_rate]") {
  AutoRate rate(6);
  std::int64_t now = 0;
  const Trace trace = run(rate, now, 60, [](std::int64_t) { return Gpu{.a = 10'000, .b = 7'000}; });
  // t(n) ≤ 0,85 × 41,7 = 35,4 ms → n = 3 (31 ms) ; ×4 coûte 38 ms.
  CHECK(rate.step() == 3);
  CHECK(changes_after(trace, 20 * kSecond) == 0);
}

TEST_CASE("auto : au pas fin, la cadence se pose près de ce que le GPU tient, sans osciller",
          "[core][auto_rate]") {
  AutoRate rate(6 * kFineGranularity, kFineGranularity);
  CHECK(rate.step() == 2 * kFineGranularity);
  std::int64_t now = 0;
  const Trace trace = run(rate, now, 90, [](std::int64_t) { return Gpu{.a = 10'000, .b = 6'000}; });
  // 10 + 6f ≤ 35,45 ms → f ≤ 4,24 : 101 / 24.
  CHECK(rate.step() >= 90);
  CHECK(rate.step() <= 104);
  CHECK(changes_after(trace, 40 * kSecond) <= 1);
  for (std::size_t i = 1; i < trace.changes.size(); ++i) {
    if (trace.changes[i].second > trace.changes[i - 1].second)
      CHECK(trace.changes[i].first - trace.changes[i - 1].first >= AutoRate::kMinRiseGapUs);
  }
  // Chaque montée est un pas mesuré (≤ 40 % + 1 : un grand saut n'est permis que loin sous la cible).
  for (std::size_t i = 1; i < trace.changes.size(); ++i) {
    if (trace.changes[i].second > trace.changes[i - 1].second)
      CHECK(trace.changes[i].second - trace.changes[i - 1].second <=
            trace.changes[i - 1].second * 40 / 100 + 1);
  }
}

TEST_CASE("auto : au pas fin, une scène lourde fait descendre puis la cadence revient", "[core][auto_rate]") {
  AutoRate rate(6 * kFineGranularity, kFineGranularity);
  std::int64_t now = 0;
  const auto gpu = [](std::int64_t t) {
    return t >= 40 * kSecond && t < 70 * kSecond ? Gpu{.a = 14'000, .b = 9'000} : Gpu{.a = 8'000, .b = 4'000};
  };
  const Trace trace = run(rate, now, 40, gpu);
  const int easy_step = rate.step();
  CHECK(easy_step >= 6 * kFineGranularity - 6);  // facile : près du maximum
  run(rate, now, 30, gpu, 0);
  CHECK(rate.step() <= 58);  // lourde : 14 + 9f ≤ 35,4 → f ≤ 2,4
  CHECK(rate.step() >= 24);
}

TEST_CASE("auto : le chien de garde descend jusqu'à ×1 quand plus aucune paire ne vient",
          "[core][auto_rate]") {
  AutoRate rate(6);
  std::int64_t now = 10 * kSecond;
  rate.seed(500, now);
  CHECK(rate.step() == 4);
  now += AutoRate::kWatchdogUs;
  CHECK(rate.tick(now, true) == 3);
  now += 1'000'000;
  CHECK(rate.tick(now, true) == 3);  // pas encore 2 s depuis la dernière descente
  for (int i = 0; i < 4; ++i) {
    now += AutoRate::kWatchdogUs;
    rate.tick(now, true);
  }
  CHECK(rate.step() == 1);
  // En pause rien ne descend, et la mesure repart.
  AutoRate paused(6);
  paused.tick(now + 60 * kSecond, false);
  CHECK(paused.tick(now + 61 * kSecond, true) == 2);
}

TEST_CASE("auto : un moteur muet ne bloque pas la descente (images manquées sans paire)",
          "[core][auto_rate]") {
  AutoRate rate(6);
  std::int64_t now = 100 * kSecond;
  rate.settle(now);  // moteur en construction : aucune paire ne vient
  for (int i = 0; i < 12; ++i) {
    now += 500'000;
    rate.missed(now, 4);  // l'écran manque des images
    rate.tick(now, true);
  }
  CHECK(rate.step() == 1);
}

TEST_CASE("auto : aucune montée avant que le moteur livre des paires", "[core][auto_rate]") {
  AutoRate rate(6);
  std::int64_t now = 0;
  run(rate, now, 30, [](std::int64_t) { return Gpu{.a = 6'000, .b = 4'000}; });
  CHECK(rate.step() == 6);
  rate.settle(now);
  now += 2'000'000;
  rate.tick(now, true);  // watchdog : un cran en dessous
  CHECK(rate.step() == 5);
  // Quelques paires seulement : pas de montée avant kMinPairs.
  for (std::uint32_t i = 0; i + 1 < AutoRate::kMinPairs; ++i) {
    now += kPeriod;
    rate.pair({.now_us = now, .made_step = rate.step(), .late = 0, .period_us = kPeriod});
  }
  CHECK(rate.step() == 5);
}

TEST_CASE("auto : un palier tenté qui échoue pose un plafond qui s'estompe et ne bloque pas",
          "[core][auto_rate]") {
  AutoRate rate(4);
  std::int64_t now = 0;
  // Tout est facile sauf ×4, qui rate : le plafond appris est ×3.
  const auto gpu = [](std::int64_t) { return Gpu{.a = 6'000, .b = 4'000}; };
  run(rate, now, 12, gpu);
  CHECK(rate.step() == 4);
  rate.settle(now);
  AutoRate fresh(4);
  std::int64_t t = 0;
  fresh.seed(300, t);
  CHECK(fresh.step() == 2);
  // Deux retards au palier tenté (poussé à ×3 par la mémoire/prédiction) posent le plafond appris.
  const std::int64_t probe_at = 20 * kSecond;
  for (int i = 0; i < 40; ++i) {
    t = probe_at + i * kPeriod;
    fresh.cost(3, 20'000, t);
    fresh.cost(2, 16'000, t);
    fresh.pair({.now_us = t, .made_step = fresh.step(), .late = 0, .period_us = kPeriod});
  }
  CHECK(fresh.step() == 3);
  t += AutoRate::kHoldUs + 10;
  fresh.pair({.now_us = t, .made_step = 3, .late = 1, .period_us = kPeriod});
  fresh.pair({.now_us = t + 1000, .made_step = 3, .late = 1, .period_us = kPeriod});
  CHECK(fresh.step() == 2);
  CHECK(fresh.learned_cap(t + 2000) == 2);
  CHECK(fresh.learned_cap(t + 40 * kSecond) == 0);  // estompé : jamais bloquant
}

TEST_CASE("auto : la coupe de scène divise le plafond appris", "[core][auto_rate]") {
  AutoRate rate(4);
  const std::int64_t t0 = 50 * kSecond;
  rate.seed(300, t0);
  for (int i = 0; i < 40; ++i) {
    rate.cost(3, 20'000, t0 + i * kPeriod);
    rate.cost(2, 16'000, t0 + i * kPeriod);
    rate.pair({.now_us = t0 + i * kPeriod, .made_step = rate.step(), .late = 0, .period_us = kPeriod});
  }
  std::int64_t t = t0 + 40 * kPeriod + AutoRate::kHoldUs;
  rate.pair({.now_us = t, .made_step = rate.step(), .late = 1, .period_us = kPeriod});
  rate.pair({.now_us = t + 1000, .made_step = 3, .late = 1, .period_us = kPeriod});
  REQUIRE(rate.learned_cap(t + 2000) == 2);
  // 6 s plus tard le plafond est à 0,5 (encore vif) ; une coupe de scène le fait passer à 0,25 puis plus
  // rien.
  CHECK(rate.learned_cap(t + 6 * kSecond) == 2);
  rate.scene_cut(t + 6 * kSecond);
  CHECK(rate.learned_cap(t + 6 * kSecond + 1) == 2);
  CHECK(rate.learned_cap(t + 8 * kSecond) == 0);
}

TEST_CASE("auto : jamais au-delà du plafond, et le palier le suit", "[core][auto_rate]") {
  AutoRate rate(3);
  std::int64_t now = 0;
  run(rate, now, 30, [](std::int64_t) { return Gpu{.a = 3'000, .b = 2'000}; });
  CHECK(rate.step() == 3);
  rate.set_ceiling(2);
  CHECK(rate.step() == 2);
  CHECK(rate.reason() == ajp::core::AutoReason::Ceiling);
}

TEST_CASE("auto : les demi-paliers montent et descendent d'une moitié", "[core][auto_rate]") {
  AutoRate rate(9, 2);  // plafond ×4,5
  CHECK(rate.granularity() == 2);
  CHECK(rate.step() == 4);  // départ ×2
  std::int64_t now = 0;
  run(rate, now, 40, [](std::int64_t) { return Gpu{.a = 8'000, .b = 4'000}; });
  CHECK(rate.step() == 9);
  rate.seed(250, now);
  CHECK(rate.step() == 4);  // ×2,5 mémorisé, un cran en dessous : ×2
  rate.missed(now, 5);
  CHECK(rate.step() == 3);
}

TEST_CASE("auto : quand le VRR se coupe, les unités passent sans dépasser", "[core][auto_rate]") {
  AutoRate rate(9, 2);
  std::int64_t now = 0;
  run(rate, now, 40, [](std::int64_t) { return Gpu{.a = 3'000, .b = 2'000}; });
  CHECK(rate.step() == 9);
  rate.set_ceiling(4, 1);
  CHECK(rate.granularity() == 1);
  CHECK(rate.step() == 4);  // ×4,5 → ×4
  rate.set_ceiling(4 * kFineGranularity, kFineGranularity);
  CHECK(rate.step() == 4 * kFineGranularity);
  CHECK(rate.centi() == 400);
  rate.set_ceiling(8, 2);
  CHECK(rate.step() == 8);
}

TEST_CASE("auto : départ mémorisé un cran en dessous, puis remontée rapide", "[core][auto_rate]") {
  AutoRate rate(6);
  std::int64_t now = 0;
  rate.seed(400, now);
  CHECK(rate.step() == 3);
  CHECK(rate.reason() == ajp::core::AutoReason::Memory);
  const Trace trace = run(rate, now, 8, [](std::int64_t) { return Gpu{.a = 6'000, .b = 4'000}; });
  CHECK(rate.step() >= 4);
  REQUIRE_FALSE(trace.changes.empty());
  CHECK(trace.changes.front().first < 2 * kSecond);
}

TEST_CASE("auto : le coût mesuré au-dessus de la période fait descendre sans retard", "[core][auto_rate]") {
  AutoRate rate(6);
  std::int64_t now = 20 * kSecond;
  rate.seed(500, now);  // départ ×4
  for (int i = 0; i < 10; ++i) {
    now += kPeriod;
    rate.cost(4, 60'000, now);
    rate.cost(3, 46'000, now);
    rate.pair({.now_us = now, .made_step = rate.step(), .late = 0, .period_us = kPeriod});
  }
  CHECK(rate.step() < 4);
  CHECK(rate.reason() == ajp::core::AutoReason::Predicted);
}

TEST_CASE("auto : après une scène lourde, la cadence revient vite près de ce que le GPU tient",
          "[core][auto_rate][reglage]") {
  AutoRate rate(6 * kFineGranularity, kFineGranularity);
  std::int64_t now = 0;
  const auto gpu = [](std::int64_t t) {
    return t >= 30 * kSecond && t < 50 * kSecond ? Gpu{.a = 16'000, .b = 11'000}
                                                 : Gpu{.a = 8'000, .b = 4'000};
  };
  run(rate, now, 30, gpu);
  run(rate, now, 20, gpu);  // la scène lourde : le palier est tombé bas
  const int low = rate.step();
  CHECK(low <= 50);
  // La scène s'allège à 50 s : en 6 s le palier doit avoir regagné au moins les deux tiers de l'écart.
  const Trace back = run(rate, now, 6, gpu);
  const int target = 6 * kFineGranularity;  // facile : le maximum de l'écran
  CHECK(rate.step() - low >= (target - low) * 2 / 3);
  // Les sauts de remontée sont rapides mais bornés (≤ 40 %), au plus un par seconde.
  for (std::size_t i = 1; i < back.changes.size(); ++i) {
    CHECK(back.changes[i].first - back.changes[i - 1].first >= AutoRate::kMinRiseGapUs);
  }
}

TEST_CASE("auto : en régime, au pas fin, au plus un changement par cinq secondes malgré un coût qui frémit",
          "[core][auto_rate][reglage]") {
  AutoRate rate(6 * kFineGranularity, kFineGranularity);
  std::int64_t now = 0;
  // Un GPU qui frémit de ±3 % autour de ce qui tient ~×3,5 (période 41,7 ms).
  const auto gpu = [](std::int64_t t) {
    const double jitter = 1.0 + 0.03 * std::sin(static_cast<double>(t) / 700'000.0);
    return Gpu{.a = 10'000.0 * jitter, .b = 7'600.0 * jitter};
  };
  run(rate, now, 40, gpu);  // montée et pose
  const Trace steady = run(rate, now, 120, gpu);
  CHECK(steady.changes.size() <= 24);  // 120 s / 5 s
  for (std::size_t i = 1; i < steady.changes.size(); ++i) {
    // Deux changements de même sens se tiennent à 5 s ; une descente de sécurité est libre.
    if (steady.changes[i].second > steady.changes[i - 1].second && steady.changes[i - 1].second > 0) {
      CHECK(steady.changes[i].first - steady.changes[i - 1].first >= AutoRate::kMinRiseGapUs);
    }
  }
}
