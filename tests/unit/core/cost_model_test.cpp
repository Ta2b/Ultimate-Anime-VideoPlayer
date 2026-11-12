#include "core/cost_model.h"

#include <cstdint>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using ajp::core::CostModel;
using Catch::Approx;

TEST_CASE("modèle de coût : a et b retrouvés à 1 %", "[core][cost_model]") {
  CostModel model;
  std::int64_t now = 1'000'000;
  for (int round = 0; round < 30; ++round) {
    for (int n = 2; n <= 6; ++n) {
      model.add(n, 3000 + 1500 * n, now);
      now += 8'000;
    }
  }
  const auto line = model.fit(now);
  REQUIRE(line.has_value());
  CHECK(line->a == Approx(3000.0).epsilon(0.01));
  CHECK(line->b == Approx(1500.0).epsilon(0.01));
  CHECK(model.spike(now) == Approx(1.0));
  CHECK(model.samples(4) == 16);
}

TEST_CASE("modèle de coût : sans mesure, pas de droite ; une coupe estompe les poids", "[core][cost_model]") {
  CostModel model;
  CHECK_FALSE(model.fit(0).has_value());
  for (int i = 0; i < 4; ++i) model.add(3, 9000, i * 10'000);
  REQUIRE(model.fit(40'000).has_value());
  model.fade(40'000);
  model.fade(40'000);
  CHECK_FALSE(model.fit(40'000).has_value());
  model.clear();
  CHECK(model.samples(3) == 0);
}

TEST_CASE("modèle de coût : après une coupe, les pointes de la scène d'avant ne comptent plus (B07-03)",
          "[core][cost_model]") {
  CostModel m;
  std::int64_t now = 1'000'000;
  for (int i = 0; i < 16; ++i) {  // un combat : une paire sur trois coûte le double
    m.add(3, i % 3 == 0 ? 30'000 : 15'000, now);
    now += 40'000;
  }
  CHECK(m.spike(now) > 1.3);
  m.fade(now);  // coupe vers un plan calme
  CHECK(m.spike(now) == Approx(1.0));
  for (int i = 0; i < 8; ++i) {
    m.add(3, 10'000, now);
    now += 40'000;
  }
  CHECK(m.spike(now) == Approx(1.0).margin(0.05));
  CHECK(m.samples(3) == 8);
}

TEST_CASE("modèle de coût : un accroc isolé ne déplace pas la moyenne (C07-08)", "[core][cost_model]") {
  // Lecture ×3 : une paire coûte 10 ms ; un alt-tab donne une paire à 300 ms. La moyenne montait vers 53 ms ;
  // l'accroc n'y entre plus qu'à trois fois la moyenne.
  CostModel model;
  std::int64_t now = 1'000'000;
  for (int i = 0; i < 24; ++i) {
    model.add(3, 10'000, now);
    now += 41'000;
  }
  model.add(3, 300'000, now);
  const auto line = model.fit(now);
  REQUIRE(line.has_value());
  const double cost = line->a + 3.0 * line->b;
  CHECK(cost < 14'000.0);
  // Une vraie hausse (un plan lourd, 25 ms par paire) passe en quelques paires.
  for (int i = 0; i < 8; ++i) {
    now += 41'000;
    model.add(3, 25'000, now);
  }
  const auto heavy = model.fit(now);
  REQUIRE(heavy.has_value());
  CHECK(heavy->a + 3.0 * heavy->b > 20'000.0);
}
