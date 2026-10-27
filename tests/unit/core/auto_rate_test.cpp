#include "core/auto_rate.h"

#include <catch2/catch_test_macros.hpp>

using ajp::core::auto_ceiling;
using ajp::core::AutoRate;

TEST_CASE("auto : le plafond suit l'écran", "[core][auto_rate]") {
  CHECK(auto_ceiling(165.0, 23.976) == 6);  // 143,86 i/s sous 165 Hz
  CHECK(auto_ceiling(120.0, 23.976) == 5);  // 119,88
  CHECK(auto_ceiling(60.0, 23.976) == 2);
  CHECK(auto_ceiling(144.0, 24.0) == 6);
  CHECK(auto_ceiling(0.0, 24.0) == 1);
}

TEST_CASE("auto : descend vite, remonte lentement", "[core][auto_rate]") {
  AutoRate rate(4);
  CHECK(rate.step() == 2);
  CHECK(rate.pair(true) == 1);
  CHECK(rate.pair(true) == 1);  // plancher
  for (std::uint32_t i = 0; i + 1 < AutoRate::kRisePairs; ++i) CHECK(rate.pair(false) == 1);
  CHECK(rate.pair(false) == 2);
}

TEST_CASE("auto : un palier qui échoue est re-tenté plus tard", "[core][auto_rate]") {
  AutoRate rate(3);
  const auto climb = [&](int target) {
    std::uint32_t pairs = 0;
    while (rate.step() < target) {
      rate.pair(false);
      ++pairs;
    }
    return pairs;
  };
  const std::uint32_t first = climb(3);
  CHECK(rate.pair(true) == 2);  // le palier tenté échoue
  CHECK(climb(3) > first);      // la reprise demande plus de paires propres
}

TEST_CASE("auto : jamais au-delà du plafond", "[core][auto_rate]") {
  AutoRate rate(3);
  for (int i = 0; i < 2000; ++i) rate.pair(false);
  CHECK(rate.step() == 3);
  rate.set_ceiling(2);
  CHECK(rate.step() == 2);
}
