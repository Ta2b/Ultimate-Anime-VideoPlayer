#include "media/flash_guard.h"

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
