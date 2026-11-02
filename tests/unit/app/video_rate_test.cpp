#include "app/play/video_rate.h"

#include <cstdint>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace ajp::app::play {
namespace {

TEST_CASE("VideoRate : 24 i/s sans image, la cadence des pts ensuite", "[app][grade]") {
  VideoRate rate;
  CHECK(rate.fps() == VideoRate::kDefaultFps);
  // RIFE à 165 Hz : une image toutes les 6 061 µs ; chaque image représentée deux fois (même pts).
  for (std::int64_t i = 0; i < 200; ++i) {
    rate.observe(i * 6'061, 1);
    rate.observe(i * 6'061, 1);
  }
  CHECK(rate.fps() == Catch::Approx(165.0).epsilon(0.01));
}

TEST_CASE("VideoRate : un seek ou un saut ne compte pas comme une cadence", "[app][grade]") {
  VideoRate rate;
  for (std::int64_t i = 0; i < 100; ++i) rate.observe(i * 41'708, 1);
  const double before = rate.fps();
  rate.observe(500'000'000, 2);  // nouvelle époque
  rate.observe(501'000'000, 2);  // un saut d'une seconde
  rate.observe(400'000'000, 2);  // en arrière
  CHECK(rate.fps() == Catch::Approx(before));
  CHECK(before == Catch::Approx(23.976).epsilon(0.01));
}

}  // namespace
}  // namespace ajp::app::play
