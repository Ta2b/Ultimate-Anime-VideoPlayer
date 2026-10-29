// Auto libre : la grille libre (core/free_grid.h) et la régulation par contre-pression (core/free_rate.h).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/free_grid.h"
#include "core/free_rate.h"
#include "core/vsync_phase.h"

using namespace ajp::core;

namespace {

SourceStamp stamp(std::uint32_t id, std::int64_t pts, std::uint32_t epoch = 0) {
  return {.id = {id}, .pts_us = pts, .epoch = epoch};
}

constexpr std::int64_t kSourcePeriod = 41'708;  // 23,976 i/s

}  // namespace

TEST_CASE("free grid : l'origine émet la source seule", "[free]") {
  FreeGrid grid;
  const CadenceStep step = grid.push(stamp(1, 0), {});
  CHECK(step.event == GridEvent::Start);
  REQUIRE(step.instants().size() == 1);
  CHECK(step.instants()[0].is_source());
  CHECK(step.instants()[0].pts_us == 0);
}

TEST_CASE("free grid : instants dans (g, d], strictement croissants, écart au moins le pas", "[free]") {
  FreeGrid grid;
  grid.push(stamp(1, 0), {});
  const FreeInputs inputs{.ready_us = INT64_MIN, .min_gap_us = 6'000, .step_us = 9'000};
  const CadenceStep step = grid.push(stamp(2, kSourcePeriod), inputs);
  REQUIRE(step.event == GridEvent::Pair);
  const auto instants = step.instants();
  REQUIRE(instants.size() == 4);  // 6, 15, 24 et 33 ms : le pas du GPU, sans sauter à la source
  std::int64_t previous = 0;
  for (std::size_t i = 0; i < instants.size(); ++i) {
    const std::int64_t pts = *instants[i].pts_us;
    CHECK(pts > previous);
    CHECK(pts <= kSourcePeriod);
    CHECK(instants[i].t.den == kSourcePeriod);
    CHECK(instants[i].t.num == pts);
    CHECK(pts - previous == (i == 0 ? 6'000 : 9'000));
    previous = pts;
  }
}

TEST_CASE("free grid : un instant tout près de la source droite devient la source", "[free]") {
  FreeGrid grid;
  grid.push(stamp(1, 0), {});
  const FreeInputs inputs{.ready_us = 40'000, .min_gap_us = 6'000, .step_us = 9'000};
  const CadenceStep step = grid.push(stamp(2, kSourcePeriod), inputs);
  REQUIRE(step.instants().size() == 1);
  CHECK(step.instants()[0].is_source());
  CHECK(*step.instants()[0].pts_us == kSourcePeriod);
  // La paire suivante repart un pas plus loin.
  const CadenceStep next = grid.push(stamp(3, 2 * kSourcePeriod), inputs);
  CHECK(*next.instants()[0].pts_us == kSourcePeriod + 9'000);
}

TEST_CASE("free grid : jamais avant l'horloge prédite", "[free]") {
  FreeGrid grid;
  grid.push(stamp(1, 0), {});
  const FreeInputs inputs{.ready_us = 20'000, .min_gap_us = 6'000, .step_us = 8'000};
  const CadenceStep step = grid.push(stamp(2, kSourcePeriod), inputs);
  for (const OutputInstant& at : step.instants()) {
    if (!at.is_source()) CHECK(*at.pts_us >= 20'000);
  }
  CHECK(step.instants().size() >= 2);
}

TEST_CASE("free grid : une paire déjà passée n'émet que la source", "[free]") {
  FreeGrid grid;
  grid.push(stamp(1, 0), {});
  const FreeInputs inputs{.ready_us = kSourcePeriod + 5'000, .min_gap_us = 6'000, .step_us = 8'000};
  const CadenceStep step = grid.push(stamp(2, kSourcePeriod), inputs);
  REQUIRE(step.instants().size() == 1);
  CHECK(step.instants()[0].is_source());
}

TEST_CASE("free grid : sans mesure, deux instants par paire ; au plus dix", "[free]") {
  FreeGrid grid;
  grid.push(stamp(1, 0), {});
  const CadenceStep two = grid.push(stamp(2, kSourcePeriod), FreeInputs{.min_gap_us = 1'000});
  CHECK(two.instants().size() == 2);
  // Un pas minuscule : la paire est bornée à kMaxInstantsPerPair instants.
  const CadenceStep many = grid.push(stamp(3, 2 * kSourcePeriod), FreeInputs{.min_gap_us = 1, .step_us = 10});
  CHECK(many.instants().size() <= kMaxInstantsPerPair);
}

TEST_CASE("free grid : le moteur borne les instants par paire et le tau reste dans ]0, 1[", "[free]") {
  for (const std::size_t limit : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{6}}) {
    FreeGrid grid;
    grid.push(stamp(1, 0), {});
    for (std::uint32_t k = 1; k <= 30; ++k) {
      const FreeInputs inputs{.min_gap_us = 1'000, .step_us = 2'000, .max_instants = limit};
      const CadenceStep step = grid.push(stamp(k + 1, k * kSourcePeriod), inputs);
      CHECK(step.instants().size() <= limit);
      CHECK(step.instants().size() >= 1);
      for (const OutputInstant& at : step.instants()) {
        const double tau = static_cast<double>(at.t.num) / static_cast<double>(at.t.den);
        CHECK(tau > 0.0);
        CHECK(tau <= 1.0);
        if (!at.is_source()) CHECK(tau < 1.0);
      }
    }
  }
}

TEST_CASE("free grid : la phase se reporte d'une paire à l'autre", "[free]") {
  FreeGrid grid;
  grid.push(stamp(1, 0), {});
  const FreeInputs inputs{.min_gap_us = 6'000, .step_us = 13'000};
  std::int64_t previous = 0;
  std::int64_t worst = 0;
  for (std::uint32_t k = 1; k <= 20; ++k) {
    const CadenceStep step = grid.push(stamp(k + 1, k * kSourcePeriod), inputs);
    for (const OutputInstant& at : step.instants()) {
      const std::int64_t pts = *at.pts_us;
      CHECK(pts > previous);
      worst = std::max(worst, pts - previous);
      previous = pts;
    }
  }
  // Aucun trou plus grand que le pas : la phase continue à travers les sources.
  CHECK(worst <= 13'000 + 13'000 / 4);
}

TEST_CASE("free grid : pts qui ne croît pas ou époque nouvelle : origine", "[free]") {
  FreeGrid grid;
  grid.push(stamp(1, 1'000), {});
  CHECK(grid.push(stamp(2, 1'000), {}).event == GridEvent::PtsNotIncreasing);
  CHECK(grid.push(stamp(3, 500, 1), {}).event == GridEvent::NewEpoch);
  CHECK(grid.push(stamp(4, 400, 0), {}).event == GridEvent::Stale);
  CHECK(grid.push(SourceStamp{.id = {5}, .pts_us = std::nullopt, .epoch = 1}, {}).event ==
        GridEvent::InvalidPts);
  CHECK(grid.flush(0).error().code == ajp::RefusalCode::SequenceOrder);
}

TEST_CASE("free rate : le pas suit le temps de service mesuré avec 10 % de marge", "[free]") {
  FreeRate rate;
  rate.set_min_gap(6'000);
  rate.reset(0);
  CHECK(rate.step_us(1'000'000) == 0);  // pas de mesure
  const std::int64_t now = 5'000'000;
  rate.cost(4, 4 * 10'000, now);
  CHECK(rate.step_us(now) == 11'000);
  // Un coût plus bas que l'écart minimal : l'écran borne.
  FreeRate fast;
  fast.set_min_gap(6'000);
  fast.cost(4, 4 * 2'000, now);
  CHECK(fast.step_us(now) == 6'000);
  // Une image seule ne dit rien.
  FreeRate lone;
  lone.cost(1, 50'000, now);
  CHECK(lone.step_us(now) == 0);
}

TEST_CASE("free rate : monte vite, descend lentement", "[free]") {
  FreeRate rate;
  const std::int64_t now = 5'000'000;
  rate.cost(2, 2 * 10'000, now);
  rate.cost(2, 2 * 20'000, now);
  CHECK(rate.info(now).service_us >= 16'000);  // une scène lourde passe presque tout de suite
  FreeRate slow;
  slow.cost(2, 2 * 20'000, now);
  slow.cost(2, 2 * 10'000, now);
  CHECK(slow.info(now).service_us >= 18'000);  // et une scène légère s'installe lentement
}

TEST_CASE("free rate : la contre-pression élargit tout de suite et ne se relâche qu'après 1 s", "[free]") {
  FreeRate rate;
  const std::int64_t t0 = 5'000'000;
  rate.cost(2, 2 * 10'000, t0);
  const std::int64_t calm = rate.step_us(t0);
  rate.missed(t0, 1);
  const std::int64_t pressed = rate.step_us(t0);
  CHECK(pressed > calm);
  CHECK(rate.step_us(t0 + 900'000) == pressed);   // rien ne bouge avant 1 s
  CHECK(rate.step_us(t0 + 4'000'000) < pressed);  // puis ça se relâche
  CHECK(rate.step_us(t0 + 60'000'000) == calm);   // jusqu'au calme
  for (int i = 0; i < 30; ++i) rate.missed(t0 + 1'000 * i, 5);
  CHECK(rate.pressure(t0 + 30'000) <= FreeRate::kPressureMax);
}

TEST_CASE("free rate : repli à 3 % d'images perdues sur 3 s", "[free]") {
  FreeRate rate;
  const std::int64_t t0 = 10'000'000;
  for (int i = 0; i < 12; ++i) rate.produced(10, t0 + i * 250'000);  // 120 images en 3 s
  rate.missed(t0 + 2'000'000, 2);
  CHECK_FALSE(rate.should_fall_back(t0 + 2'750'000));  // trop peu de pertes
  rate.missed(t0 + 2'100'000, 5);
  CHECK(rate.should_fall_back(t0 + 2'750'000));        // 7 / 120 : plus de 3 %
  CHECK_FALSE(rate.should_fall_back(t0 + 9'000'000));  // les pertes sont sorties de la fenêtre
  FreeRate fine;
  for (int i = 0; i < 12; ++i) fine.produced(100, t0 + i * 250'000);
  fine.missed(t0 + 2'000'000, 7);
  CHECK_FALSE(fine.should_fall_back(t0 + 2'750'000));  // 7 / 1200 : moins de 3 %
}

TEST_CASE("free : dix minutes simulées, GPU de 67 à 55 puis 120 i/s, sans dérive ni retard", "[free]") {
  // Horloge maître synthétique à la vitesse 1. Le GPU sert les images une à une ; la paire est préparée
  // 120 ms avant son pts (file de décodage), le retard vidéo est le temps de service d'une image.
  FreeGrid grid;
  FreeRate rate;
  const std::int64_t min_gap = 6'060;  // 165 Hz
  rate.set_min_gap(min_gap);
  rate.reset(0);
  const auto service_at = [](std::int64_t t) -> std::int64_t {
    const std::int64_t s = t / 1'000'000;
    if (s < 200) return 14'925;  // 67 i/s
    if (s < 400) return 18'182;  // 55 i/s
    return 8'333;                // 120 i/s
  };
  std::int64_t gpu_free = 0;  // instant où le GPU finit sa file
  std::int64_t last_pts = -1;
  std::int64_t images = 0;
  std::int64_t late = 0;
  std::int64_t sources = 0;
  std::vector<std::int64_t> per_phase(3, 0);
  std::int64_t pts = 0;
  grid.push(stamp(0, 0), {});
  constexpr std::int64_t kTotal = 600'000'000;
  for (std::uint32_t id = 1; pts < kTotal; ++id) {
    pts += kSourcePeriod;
    const std::int64_t now = std::max(gpu_free, pts - 120'000);  // l'horloge maître = le temps hôte
    const std::int64_t service = service_at(now);
    const std::int64_t step = rate.step_us(now);
    const FreeInputs inputs{.ready_us = now + service, .min_gap_us = min_gap, .step_us = step};
    const CadenceStep out = grid.push(stamp(id, pts), inputs);
    const auto instants = out.instants();
    REQUIRE_FALSE(instants.empty());
    for (const OutputInstant& at : instants) {
      const std::int64_t value = *at.pts_us;
      REQUIRE(value > last_pts);
      REQUIRE(value <= pts);
      REQUIRE(value > pts - kSourcePeriod);
      last_pts = value;
      // L'image est prête quand le GPU l'a servie : jamais après son instant.
      gpu_free = std::max(gpu_free, now) + service;
      if (!at.is_source() && gpu_free > value + 2'000) ++late;
      ++images;
      per_phase[static_cast<std::size_t>(std::min<std::int64_t>(now / 200'000'000, 2))]++;
    }
    rate.produced(static_cast<int>(instants.size()), now);
    if (instants.size() >= 2)
      rate.cost(static_cast<int>(instants.size()), service * static_cast<std::int64_t>(instants.size()), now);
    ++sources;
  }
  // La vidéo a gardé sa vitesse : chaque pts émis est exact et la fin du film est atteinte, sans dérive.
  CHECK(pts - last_pts < 30'000);  // la dernière image émise est à un pas de la dernière source, pas plus
  CHECK(sources == pts / kSourcePeriod);
  // Le débit suit le GPU : à peu près 1/(1,1·service) en dessous de l'écran, borné par lui au-dessus.
  const double fps0 = static_cast<double>(per_phase[0]) / 200.0;
  const double fps1 = static_cast<double>(per_phase[1]) / 200.0;
  const double fps2 = static_cast<double>(per_phase[2]) / 200.0;
  CHECK(fps0 > 50.0);
  CHECK(fps0 < 66.0);
  CHECK(fps1 > 42.0);
  CHECK(fps1 < 54.0);
  CHECK(fps2 > 80.0);
  CHECK(fps2 < 121.0);
  CHECK(late * 100 < images);  // moins de 1 % d'images terminées après leur instant
}

// ---- Écran à fréquence fixe (sans VRR) : les instants tombent sur les vsyncs ----

namespace {

constexpr std::int64_t kVsync = 16'667;  // 60 Hz

std::int64_t off_grid(std::int64_t pts, std::int64_t origin, std::int64_t period) {
  const std::int64_t r = ((pts - origin) % period + period) % period;
  return std::min(r, period - r);
}

}  // namespace

TEST_CASE("free grid vsync : instants multiples de la période d'écran à la phase près", "[free][vsync]") {
  FreeGrid grid;
  grid.push(stamp(1, 0), {});
  const std::int64_t origin = 4'321;
  const FreeInputs inputs{
      .min_gap_us = kVsync, .step_us = 12'000, .vsync_period_us = kVsync, .vsync_origin_us = origin};
  std::int64_t previous = 0;
  for (std::uint32_t k = 1; k <= 60; ++k) {
    const CadenceStep step = grid.push(stamp(k + 1, k * kSourcePeriod), inputs);
    for (const OutputInstant& at : step.instants()) {
      const std::int64_t pts = *at.pts_us;
      CHECK(pts > previous);
      CHECK(pts - previous >= kVsync - kVsync / 8);  // au moins un vsync entre deux images
      CHECK(pts <= k * kSourcePeriod);
      if (!at.is_source()) CHECK(off_grid(pts, origin, kVsync) == 0);
      previous = pts;
    }
  }
}

TEST_CASE("free grid vsync : un GPU plus lent laisse l'image tenue des vsyncs de plus", "[free][vsync]") {
  FreeGrid grid;
  grid.push(stamp(1, 0), {});
  // Un pas de 40 ms (GPU à 25 i/s) sur un écran de 60 Hz : une image tous les 40 ms en moyenne, jamais entre
  // deux vsyncs.
  const FreeInputs inputs{
      .min_gap_us = kVsync, .step_us = 40'000, .vsync_period_us = kVsync, .vsync_origin_us = 0};
  std::int64_t count = 0;
  std::int64_t last = 0;
  for (std::uint32_t k = 1; k <= 240; ++k) {
    for (const OutputInstant& at : grid.push(stamp(k + 1, k * kSourcePeriod), inputs).instants()) {
      if (!at.is_source()) CHECK(off_grid(*at.pts_us, 0, kVsync) == 0);
      ++count;
      last = *at.pts_us;
    }
  }
  const double fps = static_cast<double>(count) * 1e6 / static_cast<double>(last);
  CHECK(fps > 22.0);
  CHECK(fps < 26.0);
  // Une paire déjà passée (horloge prédite) : la source seule, l'image précédente reste tenue.
  const CadenceStep late =
      grid.push(stamp(300, 241 * kSourcePeriod), FreeInputs{.ready_us = 241 * kSourcePeriod + 1,
                                                            .min_gap_us = kVsync,
                                                            .step_us = 12'000,
                                                            .vsync_period_us = kVsync});
  REQUIRE(late.instants().size() == 1);
  CHECK(late.instants()[0].is_source());
}

TEST_CASE("vsync phase : suit la phase des présentations et rend le vsync le plus proche", "[free][vsync]") {
  VsyncPhase phase;
  CHECK(phase.near(1'000'000) == 1'000'000);  // inconnue
  phase.set_period(kVsync);
  const std::int64_t offset = 5'000;
  for (int k = 0; k < 200; ++k) phase.note(offset + k * kVsync + (k % 3 - 1) * 300);  // gigue de ±0,3 ms
  REQUIRE(phase.known());
  const std::int64_t near = phase.near(100 * kVsync + 7'000);
  CHECK(off_grid(near, offset, kVsync) <= 300);
  CHECK(std::abs(near - (100 * kVsync + 7'000)) <= kVsync / 2 + 300);
}

TEST_CASE("free vsync : dix minutes simulées, horloge qui dérive de ±0,1 %, sans dérive de phase",
          "[free][vsync]") {
  for (const double drift : {+0.001, -0.001}) {
    // Le temps de contenu avance de (1 + drift) par µs d'hôte (horloge audio contre horloge de l'écran) ; les
    // vsyncs réels sont à 3 ms + k × période (hôte), vus avec une gigue de ±0,2 ms.
    const auto media_at = [drift](std::int64_t host) {
      return static_cast<std::int64_t>(std::llround(static_cast<double>(host) * (1.0 + drift)));
    };
    const std::int64_t delay = 20'000;
    const std::int64_t true_origin = 3'000;
    FreeGrid grid;
    VsyncPhase phase;
    phase.set_period(kVsync);
    std::int64_t vsync_index = 0;  // les vsyncs notés jusqu'ici
    std::int64_t previous = -1;
    std::int64_t worst = 0;
    std::int64_t images = 0;
    std::int64_t pts = 0;
    grid.push(stamp(0, 0), {});
    for (std::uint32_t id = 1; pts < 600'000'000; ++id) {
      pts += kSourcePeriod;
      // La paire est préparée 150 ms avant son pts (hôte).
      const auto host_now = static_cast<std::int64_t>(static_cast<double>(pts) / (1.0 + drift)) - 150'000;
      while (true_origin + (vsync_index + 1) * kVsync <= host_now) {
        ++vsync_index;
        phase.note(true_origin + vsync_index * kVsync + (vsync_index % 3 - 1) * 200);
      }
      if (!phase.known()) {
        grid.push(stamp(id, pts), {});
        continue;
      }
      const std::int64_t anchor = phase.near(host_now) + delay + kVsync / 2;
      const FreeInputs inputs{
          .ready_us = media_at(host_now) + 15'000,
          .min_gap_us = kVsync,
          .step_us = 22'000,
          .vsync_period_us = static_cast<std::int64_t>(std::llround(kVsync * (1.0 + drift))),
          .vsync_origin_us = media_at(anchor)};
      for (const OutputInstant& at : grid.push(stamp(id, pts), inputs).instants()) {
        const std::int64_t value = *at.pts_us;
        REQUIRE(value > previous);
        previous = value;
        ++images;
        if (at.is_source()) continue;
        // Le vsync réel de cette image : son pts ramené à l'hôte, moins le délai et une demi-période.
        const double host =
            static_cast<double>(value) / (1.0 + drift) - static_cast<double>(delay + kVsync / 2);
        const double r =
            std::fmod(host - static_cast<double>(true_origin) + 1e12, static_cast<double>(kVsync));
        worst = std::max(worst, static_cast<std::int64_t>(std::min(r, static_cast<double>(kVsync) - r)));
      }
    }
    CHECK(images > 20'000);
    CHECK(worst < 1'000);  // jamais à plus d'1 ms de son vsync pendant dix minutes
  }
}
