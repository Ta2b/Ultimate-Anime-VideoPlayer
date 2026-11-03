#include "app/play/chain_shots.h"

#include <catch2/catch_test_macros.hpp>

namespace ajp::app::play {

TEST_CASE("Pré-analyse : aucun plan à lire pour une source HDR sans grade natif", "[app][shots]") {
  PlaySettings settings;
  settings.grade = true;
  settings.grade_values.adaptive = true;
  settings.hdr_native = false;
  ShotWork work;
  open_shots(work, "inexistant.mkv", settings, 0, true);
  CHECK(work.current.probe == nullptr);
  CHECK(work.current.timeline == nullptr);
}

}  // namespace ajp::app::play
