// mode Rapide, A12-12 : l'estimation du temps de service ne se fige plus au-dessus de l'écart des sources.
// Avant, seules les paires de deux images la mesuraient ; un à-coup du GPU la portait au-dessus d'une paire,
// plus aucune paire n'avait deux images, et l'interpolation s'arrêtait pour la séance.
#include <cstdint>

#include <catch2/catch_test_macros.hpp>

#include "core/free_grid.h"
#include "core/free_rate.h"

using namespace ajp::core;

namespace {

constexpr std::int64_t kSourcePeriod = 41'708;  // 23,976 i/s

SourceStamp stamp(std::uint32_t id, std::int64_t pts) {
  return {.id = {id}, .pts_us = pts, .epoch = 0};
}

struct Played {
  int pairs_of_two = 0;   // paires d'au moins deux images
  int lone = 0;           // paires d'une image interpolée
  int starved = 0;        // paires réduites à leur source par le pas
  std::int64_t step = 0;  // le pas à la fin
};

// Joue `seconds` de 23,976 i/s : la grille planifie sur le pas de `rate`, chaque image calculée coûte
// `image_us` au GPU. `ready_lag_us` : l'avance de l'horloge prédite sur la source gauche (GPU en retard).
Played play(FreeRate& rate, FreeGrid& grid, std::uint32_t& id, std::int64_t& now, int seconds,
            std::int64_t image_us, std::int64_t min_gap_us = 6'061, std::int64_t ready_lag_us = INT64_MIN) {
  Played out;
  const int pairs = seconds * 24;
  for (int i = 0; i < pairs; ++i) {
    ++id;
    now += kSourcePeriod;
    const std::int64_t d = static_cast<std::int64_t>(id) * kSourcePeriod;
    const FreeInputs inputs{
        .ready_us = ready_lag_us == INT64_MIN ? INT64_MIN : d - kSourcePeriod + ready_lag_us,
        .min_gap_us = min_gap_us,
        .step_us = rate.step_us(now)};
    const CadenceStep step = grid.push(stamp(id, d), inputs);
    REQUIRE(step.event == GridEvent::Pair);
    const int images = static_cast<int>(step.instants().size());
    const bool interpolated = !step.instants().front().is_source();
    if (grid.starved()) {
      ++out.starved;
      rate.starved(now);
    } else if (images >= 2 || interpolated) {
      images >= 2 ? ++out.pairs_of_two : ++out.lone;
      rate.cost(images, images * image_us, now);
    }
  }
  out.step = rate.step_us(now);
  return out;
}

struct Session {
  FreeRate rate;
  FreeGrid grid;
  std::uint32_t id = 0;
  std::int64_t now = 10'000'000;
  Session() {
    rate.set_min_gap(6'061);
    grid.push(stamp(0, 0), {});
  }
};

}  // namespace

TEST_CASE("free rate : un à-coup du GPU ne coupe pas l'interpolation pour la séance", "[free]") {
  Session s;
  play(s.rate, s.grid, s.id, s.now, 2, 10'000);
  CHECK(s.rate.step_us(s.now) == 11'000);
  // L'à-coup : une paire de deux images à 100 ms par image. Le pas dépasse l'écart des sources.
  s.rate.cost(2, 200'000, s.now);
  CHECK(s.rate.step_us(s.now) == 70'400);
  // La première seconde : des paires à une image, mesurées ou affamées, font redescendre le pas.
  const Played first = play(s.rate, s.grid, s.id, s.now, 1, 10'000);
  CHECK(first.lone + first.starved > 0);
  CHECK(first.step < kSourcePeriod);
  // Une minute plus tard : le pas d'avant l'à-coup, toutes les paires interpolées.
  const Played later = play(s.rate, s.grid, s.id, s.now, 60, 10'000);
  CHECK(later.step <= 11'100);
  CHECK(later.starved == 0);
  CHECK(later.pairs_of_two == 60 * 24);
}

TEST_CASE("free rate : un à-coup au plafond de mesure se rattrape aussi", "[free]") {
  // Le pas dépasse quatre paires : tout instant est ramené à la source (un quart de pas), aucune image
  // interpolée ne mesure plus rien. Seules les paires affamées font redescendre l'estimation.
  Session s;
  play(s.rate, s.grid, s.id, s.now, 2, 10'000);
  s.rate.cost(2, 2 * FreeRate::kMaxSampleUs, s.now);
  CHECK(s.rate.step_us(s.now) > 4 * kSourcePeriod);
  const Played after = play(s.rate, s.grid, s.id, s.now, 2, 10'000);
  CHECK(after.starved > 0);
  CHECK(after.step < kSourcePeriod / 2);
}

TEST_CASE("free rate : un GPU vraiment lent n'est pas pris pour un à-coup", "[free]") {
  // 60 ms par image interpolée : plus d'une paire. Les images seules mesurées tiennent l'estimation haute,
  // les paires affamées la sondent sans l'emballer.
  Session s;
  s.rate.cost(2, 2 * 60'000, s.now);
  const Played slow = play(s.rate, s.grid, s.id, s.now, 30, 60'000);
  CHECK(slow.pairs_of_two == 0);
  CHECK(slow.lone > 0);
  CHECK(s.rate.info(s.now).service_us >= 30'000);
}

TEST_CASE("free rate : les mesures sont ignorées pendant la reprise, la paire affamée aussi", "[free]") {
  FreeRate rate;
  const std::int64_t now = 10'000'000;
  rate.cost(2, 2 * 50'000, now);
  rate.settle(now);
  rate.starved(now + 1'000);
  rate.cost(1, 5'000, now + 1'000);
  CHECK(rate.info(now + 1'000).service_us == 50'000);
  rate.starved(now + FreeRate::kSettleUs);
  CHECK(rate.info(now + FreeRate::kSettleUs).service_us == 44'000);
}

TEST_CASE("free rate : la paire affamée en mouvement fort baisse le temps qui a fixé son écart", "[free]") {
  FreeRate rate;
  const std::int64_t now = 10'000'000;
  rate.cost(2, 2 * 10'000, now);
  rate.cost(2, 2 * 50'000, now, true);
  rate.starved(now, true);
  CHECK(rate.info(now).heavy_service_us == 44'000);
  CHECK(rate.info(now).service_us == 10'000);
  // Sans mesure lourde, l'écart vient du temps calme × kHeavyGuess : c'est lui qui redescend.
  FreeRate guess;
  guess.cost(2, 2 * 30'000, now);
  guess.starved(now, true);
  CHECK(guess.info(now).service_us == 26'400);
}

TEST_CASE("free grid : affamée par le pas seulement, pas par l'horloge ni par une coupe", "[free]") {
  // Le pas reporté dépasse la paire : la source seule, alors qu'un instant tenait.
  FreeGrid grid;
  grid.push(stamp(1, 0), {});
  const FreeInputs wide{.ready_us = INT64_MIN, .min_gap_us = 6'061, .step_us = 70'400};
  grid.push(stamp(2, kSourcePeriod), wide);  // un instant à 6 ms, l'échéance suivante à 76 ms
  CHECK_FALSE(grid.starved());
  const CadenceStep none = grid.push(stamp(3, 2 * kSourcePeriod), wide);
  REQUIRE(none.instants().size() == 1);
  CHECK(none.instants()[0].is_source());
  CHECK(grid.starved());
  // Le GPU en retard : l'horloge prédite est déjà à la source droite. Ce n'est pas le pas.
  FreeGrid late;
  late.push(stamp(1, 0), {});
  const FreeInputs behind{.ready_us = kSourcePeriod - 2'000, .min_gap_us = 6'061, .step_us = 9'000};
  const CadenceStep only = late.push(stamp(2, kSourcePeriod), behind);
  REQUIRE(only.instants().size() == 1);
  CHECK(only.instants()[0].is_source());
  CHECK_FALSE(late.starved());
  // Sans mesure du temps de service, rien à faire redescendre.
  FreeGrid unknown;
  unknown.push(stamp(1, 0), {});
  unknown.push(stamp(2, kSourcePeriod), {.ready_us = kSourcePeriod - 2'000, .min_gap_us = 6'061});
  CHECK_FALSE(unknown.starved());
  // L'origine d'une époque : jamais affamée.
  CHECK(grid.push(SourceStamp{.id = {9}, .pts_us = 0, .epoch = 1}, wide).event == GridEvent::NewEpoch);
  CHECK_FALSE(grid.starved());
}
