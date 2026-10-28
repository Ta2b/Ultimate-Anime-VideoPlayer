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

namespace {
// Deux retards rapprochés : ce qu'il faut pour descendre (un seul est toléré).
int two_lates(AutoRate& rate) {
  rate.pair(true);
  return rate.pair(true);
}
}  // namespace

TEST_CASE("auto : descend vite, remonte lentement", "[core][auto_rate]") {
  AutoRate rate(4);
  CHECK(rate.step() == 2);
  CHECK(rate.pair(true) == 2);  // un retard isolé : le palier reste
  CHECK(rate.pair(true) == 1);  // le deuxième le fait descendre
  CHECK(two_lates(rate) == 1);  // plancher (retards ignorés : temps de grâce)
  for (std::uint32_t i = 0; i < AutoRate::kGracePairs - 2; ++i) rate.pair(false);
  for (std::uint32_t i = 0; i + 1 < AutoRate::kRisePairs; ++i) CHECK(rate.pair(false) == 1);
  CHECK(rate.pair(false) == 2);
}

TEST_CASE("auto : un retard isolé ne fait pas descendre", "[core][auto_rate]") {
  AutoRate rate(4);
  for (int i = 0; i < 200; ++i) {
    for (int k = 0; k < AutoRate::kLateWindow; ++k) rate.pair(false);
    CHECK(rate.pair(true) >= 2);
  }
}

TEST_CASE("auto : les retards d'une file déjà faite ne font pas chuter en cascade", "[core][auto_rate]") {
  AutoRate rate(6);
  for (int i = 0; i < 100; ++i) rate.pair(false);
  const int before = rate.step();
  CHECK(two_lates(rate) == before - 1);
  for (std::uint32_t i = 0; i < AutoRate::kGracePairs; ++i) CHECK(rate.pair(true) == before - 1);
  CHECK(two_lates(rate) == before - 2);  // la grâce finie, de vrais retards descendent
}

TEST_CASE("auto : un saut (seek) ignore les premières paires", "[core][auto_rate]") {
  AutoRate rate(4);
  rate.settle();
  for (std::uint32_t i = 0; i < AutoRate::kGracePairs; ++i) CHECK(rate.pair(true) == 2);
  CHECK(two_lates(rate) == 1);
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
  CHECK(two_lates(rate) == 2);  // le palier tenté échoue
  CHECK(climb(3) > first);      // la reprise demande plus de paires propres
}

TEST_CASE("auto : jamais au-delà du plafond", "[core][auto_rate]") {
  AutoRate rate(3);
  for (int i = 0; i < 2000; ++i) rate.pair(false);
  CHECK(rate.step() == 3);
  rate.set_ceiling(2);
  CHECK(rate.step() == 2);
}

TEST_CASE("auto : un palier qui échoue deux fois devient un plafond appris", "[core][auto_rate]") {
  AutoRate rate(3);
  const auto climb_to_3 = [&] {
    while (rate.step() < 3) rate.pair(false);
  };
  climb_to_3();
  CHECK(two_lates(rate) == 2);
  CHECK(rate.learned_cap() == 0);
  climb_to_3();
  CHECK(two_lates(rate) == 2);
  CHECK(rate.learned_cap() == 2);
  std::uint32_t pairs = 0;
  while (rate.step() < 3 && pairs < AutoRate::kCapPairs + 100) {
    rate.pair(false);
    ++pairs;
  }
  CHECK(pairs >= AutoRate::kCapPairs - AutoRate::kGracePairs);  // re-tenté après 5 min seulement
}

TEST_CASE("auto : une cadence source absurde garde ×1", "[core][auto_rate]") {
  CHECK(auto_ceiling(165.0, 1000.0) == 1);
  CHECK(auto_ceiling(165.0, 2.0) == 1);
  CHECK(auto_ceiling(165.0, 25.0) == 6);
}

TEST_CASE("auto : les images manquées font descendre sans retard de paire", "[core][auto_rate]") {
  CHECK(ajp::core::auto_overloaded(707, 7900));       // DistilDRBA ×6
  CHECK(ajp::core::auto_overloaded(104, 7096));       // RIFE ×5 : 1,4 %
  CHECK_FALSE(ajp::core::auto_overloaded(29, 5731));  // RIFE ×4 : 0,5 %
  CHECK_FALSE(ajp::core::auto_overloaded(5, 100));    // trop peu pour conclure
  AutoRate rate(6);
  for (int i = 0; i < 100; ++i) rate.pair(false);
  const int before = rate.step();
  CHECK(rate.overloaded() == before - 1);
  CHECK(rate.overloaded() == before - 1);  // la grâce après une descente
}

TEST_CASE("auto : une tentative qui échoue sur des images manquées pose le plafond tout de suite",
          "[core][auto_rate]") {
  AutoRate rate(4);
  while (rate.step() < 3) rate.pair(false);  // tentative de ×3
  CHECK(rate.learned_cap() == 0);
  CHECK(rate.overloaded() == 2);
  CHECK(rate.learned_cap() == 2);
}
