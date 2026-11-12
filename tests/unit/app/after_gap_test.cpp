// Le tour de boucle après un trou dans le temps (audit 3, C12-09) : veille annoncée, veille vue, gel de la
// boucle sans pause, GPU refusé ou perdu au réveil.
#include "app/play/after_gap.h"

#include <cstdint>
#include <expected>
#include <optional>
#include <ranges>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "app/play/controls.h"
#include "app/play/session.h"
#include "doubles.h"
#include "french_ui.h"

namespace ajp::app::play {
namespace {

using test::FakeClock;
using test::FakeScreen;
using test::FakeStages;
using test::ring_frame;

constexpr std::int64_t kSourceUs = 41'708;
constexpr std::int64_t kGapUs = 30'000'000;
using Health = std::optional<std::expected<void, Refusal>>;

// Une session qui montre une image, l'horloge à 1 s.
struct Rig {
  FrenchUi french;
  FakeStages stages;
  FakeScreen screen;
  FakeClock clock;
  Session session{SessionConfig{.display_period_us = 16'667}, stages, screen, clock};
  Controls controls;
  Rig() {
    for (std::uint32_t k = 0; k < 40; ++k) session.handle(ring_frame(k, k * kSourceUs));
    (void)session.tick(screen.now_us);
    clock.value = {.media_us = 1'000'000, .host_us = screen.now_us, .rate = 1.0, .valid = true};
    (void)session.tick(screen.now_us);
    stages.log.clear();
  }
  bool sought_exact() const {
    return std::ranges::any_of(
        stages.log, [](const std::string& l) { return l.starts_with("seek ") && !l.ends_with("rapide"); });
  }
  bool toast() const { return controls.toast(screen.now_us).has_value(); }
};

}  // namespace

TEST_CASE("C12-09 : veille annoncée, pause et réouverture de la chaîne, sans seek", "[app][sleep]") {
  Rig rig;
  bool announced = true;
  const AfterGap out =
      after_gap_seen(rig.session, rig.controls, kGapUs, false, Health{}, announced, rig.screen.now_us);
  CHECK(out.reopen);
  CHECK_FALSE(out.lost);
  CHECK_FALSE(announced);
  CHECK(rig.session.paused());
  CHECK_FALSE(rig.sought_exact());  // la chaîne rouverte repart de la position
  CHECK(rig.toast());
}

TEST_CASE("C31-08 : veille vue au réveil seulement, pause et réouverture de la chaîne", "[app][sleep]") {
  Rig rig;
  bool announced = false;
  const AfterGap out = after_gap_seen(rig.session, rig.controls, kGapUs, true,
                                      Health{std::expected<void, Refusal>{}}, announced, rig.screen.now_us);
  CHECK(out.reopen);
  CHECK(rig.session.paused());
  CHECK_FALSE(rig.sought_exact());  // la chaîne rouverte repart de la position
  CHECK(rig.toast());
}

TEST_CASE("C12-09 : gel de la boucle, recalage sans pause ni message", "[app][sleep]") {
  Rig rig;
  bool announced = false;
  const AfterGap out =
      after_gap_seen(rig.session, rig.controls, kGapUs, false, Health{}, announced, rig.screen.now_us);
  CHECK_FALSE(out.reopen);
  CHECK_FALSE(rig.session.paused());
  CHECK(rig.sought_exact());
  CHECK_FALSE(rig.toast());
}

TEST_CASE("C12-09 : GPU refusé au réveil, la chaîne se rouvre ; perdu, la relance", "[app][sleep]") {
  {
    Rig rig;
    bool announced = false;
    const Health refused{std::unexpected(Refusal{RefusalCode::GpuMemoryExhausted, "tampons perdus"})};
    const AfterGap out =
        after_gap_seen(rig.session, rig.controls, kGapUs, true, refused, announced, rig.screen.now_us);
    CHECK(out.reopen);
    CHECK_FALSE(out.lost);
    CHECK(rig.session.paused());
    CHECK_FALSE(rig.sought_exact());
  }
  {
    Rig rig;
    bool announced = false;
    const Health lost{std::unexpected(Refusal{RefusalCode::DeviceLost, "appareil perdu"})};
    const AfterGap out =
        after_gap_seen(rig.session, rig.controls, kGapUs, true, lost, announced, rig.screen.now_us);
    REQUIRE(out.lost);
    CHECK(out.lost->code == RefusalCode::DeviceLost);
  }
}

}  // namespace ajp::app::play
