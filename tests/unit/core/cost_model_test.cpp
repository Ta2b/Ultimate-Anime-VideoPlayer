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
