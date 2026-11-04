// Le grade de l'anneau dans l'étage gpu (chantier 2.9) : la version de chaque image écrite, choisie au pts
// par LiveSettings::grade_at, confiée une fois au backend, et 0 hors du grade de l'anneau.
#include <cstdint>
#include <format>
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "app/play/live_settings.h"
#include "upscale_rig.h"

namespace ajp::app::play {
using namespace upscale_test;

TEST_CASE("Étage gpu : la version du grade de l'anneau suit le pts de chaque image écrite",
          "[app][stage][grade_ring]") {
  const bool with_rife = GENERATE(false, true);
  INFO("RIFE " << with_rife);
  Rig rig;
  rig.backend.offer_grade(true);
  rig.live->set_deband_source(true);
  const std::uint32_t version = rig.live->post_grade(GradeVersion{.epoch = 0, .from_pts_us = 100'000});
  rig.start(rig.setup(with_rife));
  rig.feed(1, 6);
  const std::size_t expected = with_rife ? 11 : 6;
  REQUIRE(rig.collect(expected));
  rig.stage.reset();

  const std::vector<RingFrame> frames = rig.sink.frames();
  const std::vector<test::RingWritten> written = rig.backend.written();
  REQUIRE(frames.size() == expected);
  REQUIRE(written.size() == expected);
  for (std::size_t i = 0; i < frames.size(); ++i) {
    INFO("image " << i << ", pts " << frames[i].pts_us);
    CHECK(frames[i].debanded);
    CHECK(frames[i].grade == (frames[i].pts_us >= 100'000 ? version : 0U));
    CHECK(written[i].grade.version == frames[i].grade);
    if (written[i].grade.version != 0) {
      CHECK(written[i].grade.pts_us == frames[i].pts_us);
      CHECK(written[i].grade.epoch == 0);
    }
  }
  CHECK(rig.backend.count(std::format("set_grade {}", version)) == 1);  // confié une fois
  CHECK(rig.reported().empty());
}

TEST_CASE("Étage gpu : sans grade dans le backend ni débandage, l'écran garde le grade",
          "[app][stage][grade_ring]") {
  const bool backend_grades = GENERATE(false, true);
  INFO("backend qui grade " << backend_grades);
  Rig rig;
  rig.backend.offer_grade(backend_grades);
  rig.live->set_deband_source(!backend_grades);  // grade sans débandage, ou débandage sans grade
  rig.live->post_grade(GradeVersion{.epoch = 0, .from_pts_us = 0});
  rig.start(rig.setup(false));
  rig.feed(1, 4);
  REQUIRE(rig.collect(4));
  rig.stage.reset();
  for (const RingFrame& f : rig.sink.frames()) CHECK(f.grade == 0);
  for (const test::RingWritten& w : rig.backend.written()) CHECK(w.grade.version == 0);
  CHECK(rig.backend.count("set_grade 1") == 0);
}

TEST_CASE("Étage gpu : un ordre « off » rend les images suivantes à l'écran", "[app][stage][grade_ring]") {
  Rig rig;
  rig.backend.offer_grade(true);
  rig.live->set_deband_source(true);
  const std::uint32_t on = rig.live->post_grade(GradeVersion{.epoch = 0, .from_pts_us = 0});
  rig.live->post_grade(GradeVersion{.epoch = 0, .from_pts_us = 160'000, .off = true});
  rig.start(rig.setup(false));
  rig.feed(1, 6);
  REQUIRE(rig.collect(6));
  rig.stage.reset();
  for (const RingFrame& f : rig.sink.frames()) {
    INFO("pts " << f.pts_us);
    CHECK(f.grade == (f.pts_us < 160'000 ? on : 0U));
  }
}

}  // namespace ajp::app::play
