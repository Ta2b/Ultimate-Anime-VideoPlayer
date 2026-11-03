#include "app/play/live_settings.h"

#include <catch2/catch_test_macros.hpp>

namespace ajp::app::play {
namespace {

// Une paire de 4 images écrites : `copies` d'entre elles sont des copies à 100 µs, les autres du calcul à 1
// ms.
std::int64_t service_after_pairs(int copies) {
  LiveSettings live;
  live.set_free(true, 1000, 0);
  std::int64_t now = 20'000'000;
  for (std::uint32_t pair = 1; pair <= 40; ++pair) {
    live.auto_expect(pair, 4, 0);
    for (int image = 0; image < 4; ++image) {
      const bool copy = image >= 4 - copies;
      const std::int64_t submitted = now;
      now += copy ? 100 : 1000;
      live.auto_written(image == 3 ? pair : pair - 1, pair, submitted, now, copy);
    }
  }
  return live.free_info(now).service_us;
}

}  // namespace

TEST_CASE("Mode Rapide : les copies ne comptent pas dans le temps de service", "[app][free]") {
  const std::int64_t with_copies = service_after_pairs(2);  // 2 images calculées à 1 ms + 2 copies
  const std::int64_t all_computed = service_after_pairs(0);
  CHECK(with_copies > 900);  // ≈ 1 000 µs : une image calculée, pas (2 000 + 200) / 4 = 550
  CHECK(with_copies < 1100);
  CHECK(all_computed > 900);
}

}  // namespace ajp::app::play
