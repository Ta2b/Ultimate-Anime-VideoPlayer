#include "core/moving_area.h"

#include <array>

#include <catch2/catch_test_macros.hpp>

namespace ajp::core {

TEST_CASE("zone qui bouge : la boîte des blocs au-dessus du seuil", "[moving_area]") {
  // 4 × 3 blocs ; deux blocs bougent, en (1, 0) et (2, 2) ; un troisième reste au seuil.
  const std::array<std::uint32_t, 12> d{0, 300, 0, 0, 0, 0, 0, 179, 0, 0, 500, 0};
  const MovingArea m = moving_area(d, 4, moving_limit(70));
  CHECK(m.blocks == 2);
  CHECK(m.total == 12);
  CHECK((m.x0 == 1 && m.x1 == 3 && m.y0 == 0 && m.y1 == 3));
  CHECK(m.box_share() == 0.5);
  const MovingArea still = moving_area(std::array<std::uint32_t, 4>{1, 2, 3, 4}, 2, 179);
  CHECK((still.blocks == 0 && still.box_share() == 0.0));
}

}  // namespace ajp::core
