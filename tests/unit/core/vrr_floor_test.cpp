#include "core/vrr_floor.h"

#include <catch2/catch_test_macros.hpp>

namespace ajp::core {

TEST_CASE("VRR : une image tenue 83 ms est représentée 4 fois, à intervalles égaux", "[core][vrr]") {
  const std::int64_t floor = vrr_floor_us(48);
  CHECK(floor == 18'939);
  const std::int64_t step = floor_step_us(83'000, floor);
  CHECK(step > 16'000);
  CHECK(step < 17'200);
  CHECK(83'000 / step == 5);  // 5 intervalles : la présentation d'origine et 4 de plus
  CHECK(step <= floor);
}

TEST_CASE("VRR : au plancher ou en dessous, ou sans plancher, rien n'est représenté", "[core][vrr]") {
  CHECK(floor_step_us(18'000, 18'939) == 18'000);
  CHECK(floor_step_us(18'939, 18'939) == 18'939);
  CHECK(floor_step_us(100'000, 0) == 100'000);
  CHECK(vrr_floor_us(0) == 0);
}

}  // namespace ajp::core
