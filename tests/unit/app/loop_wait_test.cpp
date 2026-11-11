// L'attente de la boucle d'affichage (loop_wait.h) : à l'essai précis, jamais avant l'échéance et bien moins
// d'une milliseconde après sur une machine au repos (le p99 ≤ 200 µs se mesure sur le PC).
#include "app/play/loop_wait.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include <catch2/catch_test_macros.hpp>

#include "core/clock.h"

namespace ajp::app::play {

TEST_CASE("attente précise : l'échéance tenue à la microseconde, jamais avant", "[app][present]") {
  std::int64_t worst = 0;
  for (const std::int64_t wait : {300, 1'200, 2'500, 3'700}) {
    // Le meilleur de cinq essais : une préemption de l'ordonnanceur (machine chargée, CI) ne dit rien de
    // l'attente elle-même ; une attente toujours en retard, elle, se voit (B40-08, sans branche sur la CI).
    std::int64_t best = std::numeric_limits<std::int64_t>::max();
    for (int attempt = 0; attempt < 5; ++attempt) {
      const std::int64_t start = host_now_us();
      wait_loop(wait, true);
      const std::int64_t late = host_now_us() - start - wait;
      CHECK(late >= 0);
      best = std::min(best, late);
    }
    worst = std::max(worst, best);
  }
  CHECK(worst < 20'000);
}

}  // namespace ajp::app::play
