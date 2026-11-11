#include "media/flash_guard.h"

#include <array>
#include <cstdint>
#include <optional>

#include <catch2/catch_test_macros.hpp>

namespace {

// Deux secondes d'alternance noir et blanc à `hz` flashs par seconde, à 60 images par seconde ; vrai si le
// plafond a tenu à un moment.
bool limited_at(double hz) {
  ajp::media::FlashGuard guard;
  bool limited = false;
  for (int i = 0; i < 120; ++i) {
    const double t = i / 60.0;
    const bool white = static_cast<int>(t * hz * 2.0) % 2 == 1;
    limited = guard.push(i * 16'667, white ? 203.0 : 0.0) || limited;
  }
  return limited;
}

}  // namespace

TEST_CASE("flashs : plafond au-delà de trois par seconde (BT.1702-2, P3.51)", "[flash_guard]") {
  CHECK_FALSE(limited_at(3.0));
  CHECK(limited_at(6.0));
  CHECK(limited_at(10.0));
  // Une variation faible ne compte pas, même rapide.
  ajp::media::FlashGuard guard;
  bool limited = false;
  for (int i = 0; i < 120; ++i) limited = guard.push(i * 16'667, i % 2 == 0 ? 100.0 : 110.0) || limited;
  CHECK_FALSE(limited);
}

namespace {

using ajp::media::FlashGuard;
using ajp::media::kFlashZones;

// `lit` zones alternent entre low et high à hz flashs par seconde, les autres restent à low ; 60 i/s.
// Rend l'instant du premier plafond, en microsecondes.
std::optional<std::int64_t> first_limit(FlashGuard& guard, int lit, float low, float high, double hz,
                                        bool starts_high = false) {
  for (int i = 0; i < 180; ++i) {
    const bool on = (static_cast<int>(i / 60.0 * hz * 2.0) % 2 == 1) != starts_high;
    std::array<float, kFlashZones> zones{};
    zones.fill(low);
    for (int z = 0; z < lit; ++z) zones.at(static_cast<std::size_t>(z)) = on ? high : low;
    if (guard.push(i * std::int64_t{16'667}, zones)) return i * std::int64_t{16'667};
  }
  return std::nullopt;
}

}  // namespace

TEST_CASE("flashs : un quart de l'écran suffit, moins ne compte pas (A42-08)", "[flash_guard]") {
  // Un quart de l'écran de 20 à 60 cd/m², quatre fois par seconde : la moyenne ne bouge que de 10.
  FlashGuard quarter;
  CHECK(first_limit(quarter, 4, 20.0F, 60.0F, 4.0));
  FlashGuard less;
  CHECK_FALSE(first_limit(less, 3, 20.0F, 60.0F, 4.0));
  // L'image la plus sombre au-dessus de 160 cd/m² : pas un flash.
  FlashGuard bright;
  CHECK_FALSE(first_limit(bright, 16, 170.0F, 200.0F, 6.0));
}

TEST_CASE("flashs : la lumière jugée est celle que le grade montre", "[flash_guard]") {
  // 8 cd/m² de variation dans la source : rien sans grade, un flash quand le grade la porte à 20.
  FlashGuard plain;
  CHECK_FALSE(first_limit(plain, 16, 20.0F, 28.0F, 5.0));
  FlashGuard graded;
  graded.set_gain(2.5);
  CHECK(first_limit(graded, 16, 20.0F, 28.0F, 5.0));
  graded.reset();  // le gain survit à un saut
  CHECK(first_limit(graded, 16, 20.0F, 28.0F, 5.0));
}

TEST_CASE("flashs : la première variation compte, montée ou descente (A42-17)", "[flash_guard]") {
  // À 4 Hz, une transition tous les huitièmes de seconde : la septième à 875 ms, d'où qu'on parte (le départ
  // sur le noir attendait 1 000 ms, sa première montée n'étant pas comptée).
  for (const bool starts_high : {false, true}) {
    CAPTURE(starts_high);
    FlashGuard guard;
    const auto at = first_limit(guard, 16, 0.0F, 203.0F, 4.0, starts_high);
    REQUIRE(at);
    CHECK(*at <= 900'000);  // 875 ms, à une image près
    // Le début de la séquence : sa première transition, que la pré-analyse marque dans la chronologie.
    REQUIRE(guard.danger_from());
    CHECK(*guard.danger_from() <= 140'000);
    CHECK(guard.limited_until() == *at + ajp::media::kFlashHoldUs);
  }
  FlashGuard slow;
  CHECK_FALSE(first_limit(slow, 16, 0.0F, 203.0F, 3.0));
}

TEST_CASE("flashs : une analyse perdue pendant une suite de variations compte (A42-18)", "[flash_guard]") {
  // 3 Hz : six transitions par seconde, sous le seuil. Une image perdue au milieu peut en cacher deux.
  FlashGuard guard;
  bool limited = false;
  for (int i = 0; i < 120; ++i) {
    const std::int64_t pts = i * std::int64_t{16'667};
    if (i == 75) {  // six transitions dans la seconde qui précède
      limited = guard.missed(pts) || limited;
      continue;
    }
    limited = guard.push(pts, static_cast<int>(i / 60.0 * 6.0) % 2 == 1 ? 203.0 : 0.0) || limited;
  }
  CHECK(limited);
  // Hors de toute variation, une image perdue ne déclenche rien.
  FlashGuard calm;
  for (int i = 0; i < 60; ++i) CHECK_FALSE(calm.missed(i * std::int64_t{16'667}));
}

TEST_CASE("flashs : une image fixe ne se plafonne pas quand des analyses manquent", "[flash_guard]") {
  // Le début d'une vidéo (noir puis une mire fixe : une transition), puis une analyse sur deux perdue, comme
  // sous un GPU chargé. Chaque image perdue comptait pour une transition : le grade se plafonnait à 250 nits.
  FlashGuard guard;
  guard.set_gain(3.4);
  bool limited = false;
  for (int i = 0; i < 96; ++i) {
    const std::int64_t pts = i * std::int64_t{41'708};
    if (i > 1 && i % 2 == 0) {
      limited = guard.missed(pts) || limited;
      continue;
    }
    limited = guard.push(pts, i == 0 ? 0.0 : 120.0) || limited;
  }
  CHECK_FALSE(limited);
  // Une suite d'images perdues sans analyse du tout : rien non plus.
  FlashGuard lost;
  lost.push(0, 0.0);
  lost.push(41'708, 120.0);
  for (int i = 2; i < 48; ++i) CHECK_FALSE(lost.missed(i * std::int64_t{41'708}));
}
