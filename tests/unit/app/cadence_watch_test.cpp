// La cadence haute (cadence_watch.h), en un seul cas : les pertes de l'affichage restent hors de la
// contre-pression (sauf --display-loss-pressure), l'image vue prête à un tour d'avant, les modes de
// présentation et la ligne de fin qui dit ce qui a borné la cadence.
#include "app/play/cadence_watch.h"

#include <catch2/catch_test_macros.hpp>

namespace kirari::app::play {
namespace {

TEST_CASE("Cadence haute : pertes de l'affichage, images prêtes, bornes", "[app][cadence]") {
  CHECK(parse_swap_mode("mailbox") == SwapMode::Mailbox);
  CHECK(parse_swap_mode("relaxed") == SwapMode::Relaxed);
  CHECK_FALSE(parse_swap_mode("vsync"));
  CHECK(name(SwapMode::Immediate) == "immediate");

  // Trois sautées dont deux prêtes à temps, trois trous dans les pts : une seule perte du GPU.
  CHECK(pressure_losses({.missing = 3, .dropped = 3, .dropped_ready = 2}, false) == 1);
  CHECK(pressure_losses({.missing = 3, .dropped = 3, .dropped_ready = 2}, true) == 3);
  CHECK(pressure_losses({.missing = 0, .dropped = 2, .dropped_ready = 2}, false) == 0);

  ReadySeen seen;
  seen.note({1, 1, 10}, 1'000);
  seen.note({1, 1, 10}, 2'000);  // la première fois seulement
  CHECK(seen.before({1, 1, 10}, 2'000));
  CHECK_FALSE(seen.before({1, 1, 10}, 1'000));  // écrite à ce tour-ci : arrivée en retard
  CHECK_FALSE(seen.before({1, 2, 10}, 5'000));  // inconnue : comptée comme avant
  CHECK_FALSE(seen.before({2, 1, 10}, 5'000));  // anneau recréé : mêmes slot et valeur, autre image
  for (std::uint64_t v = 0; v < 40; ++v) seen.note({1, 3, v}, 3'000);  // l'anneau des 32 dernières
  CHECK_FALSE(seen.before({1, 1, 10}, 9'000));

  CadenceWatch watch({.swap_mode = SwapMode::Mailbox});
  CHECK(watch.take_losses({.missing = 4, .dropped = 4, .dropped_ready = 4}) == 0);
  CHECK(watch.take_losses({.missing = 6, .dropped = 5, .dropped_ready = 4}) == 2);  // deltas
  CHECK(watch.summary().find("mode Rapide inactif") != std::string::npos);
  // Écart minimal 6,1 ms : un relevé borné par l'écran, un par le service, deux par la contre-pression.
  watch.sample(true, {.fps = 163.0, .step_us = 6'100, .service_us = 3'800, .pressure = 1.0}, 6'100);
  watch.sample(true, {.fps = 120.0, .step_us = 8'000, .service_us = 7'000, .pressure = 1.0}, 6'100);
  watch.sample(true, {.fps = 107.0, .step_us = 9'300, .service_us = 4'200, .pressure = 2.0}, 6'100);
  watch.sample(true, {.fps = 110.0, .step_us = 9'000, .service_us = 4'100, .pressure = 1.9}, 6'100);
  watch.sample(false, {.fps = 50.0, .step_us = 1, .service_us = 1}, 6'100);  // mode inactif : ignoré
  const std::string text = watch.summary();
  CHECK(text.find("présentation mailbox") != std::string::npos);
  CHECK(text.find("pertes de l'affichage 4 (hors de la contre-pression)") != std::string::npos);
  CHECK(text.find("pertes du GPU 2") != std::string::npos);
  CHECK(text.find("borné par l'écran 25 %, par le temps de service 25 %, par la contre-pression 50 %") !=
        std::string::npos);
}

}  // namespace
}  // namespace kirari::app::play
