#include "app/play/chain_setup.h"

#include <catch2/catch_test_macros.hpp>

namespace ajp::app::play {
namespace {

PlaySettings fast_settings() {
  PlaySettings s;
  s.rife_factor = 2;
  s.rife_auto = true;
  s.auto_free = true;
  return s;
}

}  // namespace

TEST_CASE("Rapide : le warp et l'anneau suivent l'écran, pas le facteur demandé", "[app][chain]") {
  PlaySettings s = fast_settings();
  CHECK(top_factor(s, 165.0, 23.976) == 6);
  CHECK(ring_depth(s, factor_of(s)) == 8U);

  s.auto_free = false;  // Stable : le facteur demandé borne toujours
  CHECK(top_factor(s, 165.0, 23.976) == 2);
  CHECK(ring_depth(s, factor_of(s)) == 4U);
}

TEST_CASE("Rapide : jamais moins de 1, aucune interpolation sans facteur", "[app][chain]") {
  PlaySettings s = fast_settings();
  CHECK(top_factor(s, 24.0, 23.976) == 1);
  s.rife_factor = 0;
  CHECK(top_factor(s, 165.0, 23.976) == 1);
}

TEST_CASE("Repli du Rapide : la séance passe en Stable, le choix des réglages reste", "[app][settings]") {
  PlaySettings s;
  s.auto_free = true;
  CHECK(free_requested(s, false));
  CHECK_FALSE(free_requested(s, true));
  CHECK(s.auto_free);  // play.toml garde Rapide
  s.auto_free = false;
  CHECK_FALSE(free_requested(s, false));
}

}  // namespace ajp::app::play
