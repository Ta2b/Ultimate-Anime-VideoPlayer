#include "app/play/live_settings.h"

#include <cmath>

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

TEST_CASE("Mode Rapide : la grille des vblanks mesurée prime sur la phase estimée", "[app][free][present]") {
  LiveSettings live;
  live.set_free(true, 6061, 0);
  live.set_vsync(6061, 3'000);  // 165 Hz annoncés, 3 ms entre présentation et affichage
  live.vsync_note(1'000'000);
  const LiveSettings::VsyncGrid estimated = live.vsync_grid(1'000'500);
  CHECK(estimated.period_us == 6061);
  // 164,78 Hz mesurés, vblanks affichés à 2 000 000 + k × 6068,7 µs.
  const core::VblankGrid measured{.phase_us = 2'000'000.0, .period_us = 1e6 / 164.78};
  live.vsync_note(1'006'061, measured);
  const LiveSettings::VsyncGrid grid = live.vsync_grid(2'000'000 + 10 * 6069 + 1'000);
  CHECK(grid.period_us == 6069);
  CHECK(grid.anchor_host_us == std::llround(measured.at(10)) + 6069 / 2);
  // Sans grille mesurée (niveau Open), la phase estimée revient.
  live.vsync_note(1'012'122);
  CHECK(live.vsync_grid(1'000'500).period_us == 6061);
  // VRR : aucune grille, mesurée ou non.
  live.set_vsync(0, 0);
  live.vsync_note(1'018'183, measured);
  CHECK(live.vsync_grid(1'000'500).period_us == 0);
}

TEST_CASE("Rapide à pas constant (7.3) : le temps GPU de l'écran devient d, lissé", "[app][free]") {
  LiveSettings live;
  live.vsync_note(1'000'000, std::nullopt, 1'600);  // éteint : rien n'est retenu
  live.set_steady(true, {.vrr = true, .hz = 165.0}, 24.0);
  CHECK(live.steady_info().display_us == 0);
  live.vsync_note(1'000'000, std::nullopt, 1'600);
  CHECK(live.steady_info().display_us == 1'600);
  live.vsync_note(1'006'000, std::nullopt, 0);  // inconnu : ignoré
  CHECK(live.steady_info().display_us == 1'600);
  live.vsync_note(1'012'000, std::nullopt, 2'600);  // lissé à 0,1
  CHECK(live.steady_info().display_us == 1'700);
}

}  // namespace ajp::app::play
