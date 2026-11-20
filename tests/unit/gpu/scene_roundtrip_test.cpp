#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/cuda/backend/scene_roundtrip_read.h"

namespace kirari::gpu::cuda {

TEST_CASE("Le retour NVOF garde une seule paire jusqu'à sa relecture", "[gpu][nvof]") {
  RoundtripTicket slot;
  CHECK_FALSE(slot.accepts(0));
  REQUIRE(slot.accepts(12));
  slot.start(12);
  CHECK(slot.pending() == 12);
  CHECK_FALSE(slot.accepts(13));  // un événement non prêt garde le tampon
  CHECK_FALSE(slot.accepts(12));
  CHECK(slot.take() == 12);
  CHECK(slot.take() == 0);
  CHECK_FALSE(slot.accepts(12));  // une seconde mesure de la même paire est refusée
  CHECK(slot.accepts(14));        // une paire a pu être sautée
}

TEST_CASE("Le retour NVOF lit les deux champs S10.5 et écarte les projections hors champ", "[gpu][nvof]") {
  constexpr RoundtripShape odd{65, 49};
  static_assert(odd.cols() == 17 && odd.rows() == 13 && odd.words() == 442);
  CHECK(roundtrip_cell(64.0F, odd.width, odd.cols()) == 16.0F);
  constexpr RoundtripShape even{64, 48};
  std::vector<std::int16_t> forward(even.words(), 0);
  std::vector<std::int16_t> reverse(even.words(), 0);
  for (std::size_t i = 0; i < even.words(); i += 2) {
    forward[i] = 512;  // +16 pixels dans l'aller
    reverse[i] = -512;
  }
  REQUIRE_FALSE(check_roundtrip(odd, forward, reverse));
  const auto closed = check_roundtrip(even, forward, reverse);
  REQUIRE(closed);
  CHECK(closed->outside == 3);
  CHECK(closed->checked == 9);
  CHECK(closed->open == 0);
  const auto open = check_roundtrip(even, forward, forward);
  REQUIRE(open);
  CHECK(open->open == open->checked);
}

}  // namespace kirari::gpu::cuda
