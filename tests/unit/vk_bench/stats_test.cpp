// Statistiques du banc : médiane (paire et impaire), p95 au rang le plus proche, écart type de
// l'échantillon, débit effectif, et refus des séries vides ou des durées invalides.
#include "stats.h"

#include <cmath>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using ajp::RefusalCode;
using ajp::bench::gigabytes_per_second;
using ajp::bench::summarize;
using Catch::Approx;

TEST_CASE("summarize : médiane, extrêmes et écart type d'une série impaire") {
  const std::vector<double> samples{3.0, 1.0, 2.0};
  const auto s = summarize(samples);
  REQUIRE(s);
  CHECK(s->count == 3);
  CHECK(s->median == 2.0);
  CHECK(s->min == 1.0);
  CHECK(s->max == 3.0);
  CHECK(s->p95 == 3.0);
  CHECK(s->stddev == Approx(1.0));
}

TEST_CASE("summarize : médiane d'une série paire, moyenne des deux valeurs centrales") {
  const std::vector<double> samples{4.0, 1.0, 3.0, 2.0};
  const auto s = summarize(samples);
  REQUIRE(s);
  CHECK(s->median == 2.5);
  CHECK(s->stddev == Approx(std::sqrt(5.0 / 3.0)));
  CHECK(s->slow == 1);  // 4 > 1,5 × 2,5
}

TEST_CASE("summarize : p95 au rang le plus proche") {
  for (const auto& [n, want] : {std::pair{20, 19.0}, std::pair{50, 48.0}, std::pair{100, 95.0}}) {
    std::vector<double> samples(static_cast<std::size_t>(n));
    std::iota(samples.rbegin(), samples.rend(), 1.0);  // n, n − 1, ..., 1 : l'ordre ne compte pas
    const auto s = summarize(samples);
    REQUIRE(s);
    INFO("n = " << n);
    CHECK(s->p95 == want);
  }
}

TEST_CASE("summarize : une seule valeur, écart type nul") {
  const std::vector<double> samples{0.25};
  const auto s = summarize(samples);
  REQUIRE(s);
  CHECK(s->median == 0.25);
  CHECK(s->p95 == 0.25);
  CHECK(s->stddev == 0.0);
  CHECK(s->slow == 0);
}

TEST_CASE("summarize : série vide, durée négative ou non finie refusées") {
  CHECK(summarize({}).error().code == RefusalCode::FieldOutOfRange);
  const std::vector<double> negative{1.0, -0.5};
  CHECK(summarize(negative).error().code == RefusalCode::FieldOutOfRange);
  const std::vector<double> nan{1.0, std::numeric_limits<double>::quiet_NaN()};
  CHECK(summarize(nan).error().code == RefusalCode::FieldOutOfRange);
}

TEST_CASE("gigabytes_per_second : 10⁹ octets par seconde, durée en millisecondes") {
  CHECK(*gigabytes_per_second(1'000'000'000, 1000.0) == Approx(1.0));
  CHECK(*gigabytes_per_second(504'000'000, 1.0) == Approx(504.0));
  CHECK(gigabytes_per_second(1, 0.0).error().code == RefusalCode::FieldOutOfRange);
  CHECK(gigabytes_per_second(1, -1.0).error().code == RefusalCode::FieldOutOfRange);
}
