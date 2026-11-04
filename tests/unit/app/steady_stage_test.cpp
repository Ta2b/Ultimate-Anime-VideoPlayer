// Rapide à pas constant dans l'étage gpu (--fast-steady, chantier 7.7) : N images par source, constant,
// instants uniformes ; à N = 1, chaque paire est mesurée (les coupes changent N) sans rien interpoler.
#include <chrono>
#include <cstdint>
#include <format>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/live_settings.h"
#include "upscale_rig.h"

namespace ajp::app::play {
using namespace upscale_test;

TEST_CASE("Étage gpu : Rapide à pas constant, à N = 1 chaque paire est mesurée", "[app][stage][steady]") {
  Rig rig;
  rig.live->set_free(true, 6'060, host_now_us());
  rig.live->set_steady(true, {.vrr = true, .hz = 165.0}, 25.0);  // sans a priori : N part de 1
  REQUIRE(rig.live->steady_active());
  rig.start(rig.setup(true));
  rig.feed(1, 6);
  REQUIRE(rig.collect(6));
  rig.stage.reset();

  CHECK(shape_of(rig.sink.frames()) == std::vector<std::string>{"1", "2", "3", "4", "5", "6"});
  CHECK(rig.backend.count("engine rife") == 0);
  std::size_t measured = 0;
  for (std::uint32_t k = 1; k < 6; ++k) {
    measured += rig.backend.count(std::format("scores_now {} {}", k, k + 1)) +
                rig.backend.count(std::format("scores_start {} {}", k, k + 1));
  }
  CHECK(measured == 5);
}

TEST_CASE("Étage gpu : Rapide à pas constant, N = 6,5 et des instants uniformes", "[app][stage][steady]") {
  Rig rig;
  rig.live->set_free(true, 6'060, host_now_us());
  // Un GPU qui tient bien plus : N au plafond de l'écran, 165 / 25 → 6,5.
  rig.live->set_steady(true, {.vrr = true, .hz = 165.0}, 25.0, {.a = 500.0, .b = 500.0});
  REQUIRE(rig.live->steady_factor() == core::Ratio{13, 2});
  rig.start(rig.setup(true));
  rig.feed(1, 5);
  // La source 1 seule (origine), puis 4 paires × 6,5 = 26 images.
  (void)rig.collect(27, true, std::chrono::seconds(3));
  rig.stage.reset();
  INFO("forme " << [&] {
    std::string all;
    for (const std::string& f : shape_of(rig.sink.frames())) all += f + " ";
    return all;
  }());

  const std::vector<std::int64_t> pts = pts_of(rig.sink.frames());
  REQUIRE(pts.size() == 27);
  for (std::size_t i = 1; i < pts.size(); ++i) {
    INFO("image " << i);
    const std::int64_t gap = pts[i] - pts[i - 1];
    CHECK(gap >= 6'153);  // 40 000 / 6,5 = 6 153,8
    CHECK(gap <= 6'154);
  }
  CHECK(rig.backend.count("engine rife") > 0);
}

}  // namespace ajp::app::play
