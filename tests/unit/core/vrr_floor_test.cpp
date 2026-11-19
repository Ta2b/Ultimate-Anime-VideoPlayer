#include "core/vrr_floor.h"

#include <catch2/catch_test_macros.hpp>

namespace kirari::core {

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
  // Jamais plus serré que l'écran (165 Hz : 6 061 µs) : une image tenue 10,4 ms sous un plancher de 7 ms
  // ne se représente pas ; tenue 14 ms, une fois, à 7 ms.
  CHECK(floor_step_us(10'400, 7'000, 6'061) == 10'400);
  CHECK(floor_step_us(14'000, 7'000, 6'061) == 7'000);
  CHECK(floor_step_us(83'000, vrr_floor_us(48), 6'061) == floor_step_us(83'000, vrr_floor_us(48)));
}

}  // namespace kirari::core
