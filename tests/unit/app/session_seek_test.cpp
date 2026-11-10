// La session, le pas d'image et les commandes d'horloge perdues : pas bloqué en VFR (A13-05), Play et Pause
// refusés par une boîte pleine (A13-06), reprise après des pas pris dans la file (A13-07), seek perdu par la
// source (A13-08).
#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/session.h"
#include "doubles.h"
#include "french_ui.h"

namespace ajp::app::play {
namespace {

using test::FakeClock;
using test::FakeScreen;
using test::FakeStages;
using test::ring_frame;

struct SourceRate final : StatsSource {
  std::int64_t period_us = 41'667;
  std::optional<std::string> rife() const override { return std::nullopt; }
  std::optional<std::string> memory() override { return std::nullopt; }
  std::int64_t source_period_us() const override { return period_us; }
};

struct Rig {
  FakeStages stages;
  FakeScreen screen;
  FakeClock clock;
  Session session;
  explicit Rig(SessionConfig config = {}) : session(std::move(config), stages, screen, clock) {}
  void run_clock(std::int64_t media_us, std::uint32_t epoch = 0) {
    clock.value = {
        .media_us = media_us, .host_us = screen.now_us, .rate = 1.0, .epoch = epoch, .valid = true};
  }
  void tick() { REQUIRE(session.tick(screen.now_us)); }
  std::int64_t shown_pts() const {
    REQUIRE(!screen.shown.empty());
    REQUIRE(screen.shown.back().frame);
    return screen.shown.back().frame->pts_us;
  }
  std::ptrdiff_t count(const std::string& line) const { return std::ranges::count(stages.log, line); }
};

}  // namespace

// La cadence annoncée (59,94 i/s) est plus haute que la réelle (23,976 i/s) : la cible d'un pas tombe dans
// l'image montrée, le seek la rend telle quelle. Le pas suivant part de la cible, et l'image finit par
// changer.
TEST_CASE("Image par image en VFR : un pas retombé dans l'image montrée ne bloque pas le suivant (A13-05)",
          "[app][session]") {
  ajp::app::play::FrenchUi french;
  SourceRate rate;
  rate.period_us = 16'683;
  Rig rig(SessionConfig{.duration_us = 60'000'000, .extra = &rate});
  rig.session.handle(ring_frame(0, 1'000'000));
  rig.tick();
  rig.session.command(cmd::StepFrame{.direction = 1});
  CHECK(rig.stages.log.back() == "seek 1025024");
  rig.session.handle(ring_frame(1, 1'000'000, 1));  // la même image, qui dure 41,7 ms
  rig.tick();
  REQUIRE(rig.shown_pts() == 1'000'000);
  rig.session.command(cmd::StepFrame{.direction = 1});
  CHECK(rig.stages.log.back() == "seek 1041707");  // au-delà de la fin de l'image montrée
  rig.session.handle(ring_frame(2, 1'041'708, 2));
  rig.tick();
  CHECK(rig.shown_pts() == 1'041'708);
  // Partie d'une autre image, la marque est oubliée : le pas suivant repart de l'image montrée.
  rig.session.command(cmd::StepFrame{.direction = 1});
  CHECK(rig.stages.log.back() == "seek 1066732");
}

TEST_CASE("Une pause refusée par la boîte audio est redite au tour suivant (A13-06)", "[app][session]") {
  ajp::app::play::FrenchUi french;
  Rig rig;
  rig.session.handle(ring_frame(0, 40'000));
  rig.tick();
  rig.run_clock(40'000);
  rig.stages.log.clear();
  rig.stages.refuse_clock = 1;
  rig.session.command(cmd::Pause{});
  CHECK(rig.stages.log == std::vector<std::string>{"pause refusée"});
  rig.tick();
  CHECK(rig.stages.log == std::vector<std::string>{"pause refusée", "pause"});
  rig.tick();
  CHECK(rig.count("pause") == 1);  // passée : plus rien à redire
}

TEST_CASE("Un Play refusé est redit, et oublié si la pause revient d'abord (A13-06)", "[app][session]") {
  ajp::app::play::FrenchUi french;
  Rig rig;
  rig.session.handle(ring_frame(0, 40'000));
  rig.tick();
  rig.session.command(cmd::Pause{});
  rig.stages.log.clear();
  rig.stages.refuse_clock = 1;
  rig.session.command(cmd::Play{});
  CHECK(rig.stages.log == std::vector<std::string>{"play 0 40000 refusé"});
  rig.tick();
  CHECK(rig.stages.log.back() == "play 0 40000");
  rig.stages.log.clear();
  rig.stages.refuse_clock = 1;
  rig.session.command(cmd::Pause{});
  rig.session.command(cmd::Play{});  // la pause refusée est dépassée : seule la reprise compte
  rig.tick();
  CHECK(rig.count("play 0 40000") == 1);
}

TEST_CASE("Reprise après des pas pris dans la file : un seek exact recale l'audio (A13-07)",
          "[app][session]") {
  ajp::app::play::FrenchUi french;
  Rig rig(SessionConfig{.duration_us = 60'000'000});
  rig.session.handle(ring_frame(0, 1'000'000));
  rig.tick();
  rig.session.command(cmd::Pause{});
  rig.session.handle(ring_frame(1, 1'041'667));
  rig.session.command(cmd::StepFrame{.direction = 1});
  rig.tick();
  REQUIRE(rig.shown_pts() == 1'041'667);
  rig.stages.log.clear();
  rig.session.command(cmd::Play{});
  CHECK(rig.stages.log == std::vector<std::string>{"pause", "flush 1", "seek 1041667"});
  CHECK_FALSE(rig.session.paused());
  // Sans pas dans la file, la reprise part de l'image montrée, sans seek.
  rig.session.handle(ring_frame(2, 1'041'667, 1));
  rig.tick();
  rig.session.command(cmd::Pause{});
  rig.stages.log.clear();
  rig.session.command(cmd::Play{});
  CHECK(rig.stages.log == std::vector<std::string>{"play 1 1041667"});
}

TEST_CASE("Un seek perdu par la source est rejoué, de plus en plus lentement (A13-08)", "[app][session]") {
  ajp::app::play::FrenchUi french;
  Rig rig(SessionConfig{.duration_us = 60'000'000});
  rig.session.handle(ring_frame(0, 1'000'000));
  rig.tick();
  rig.session.command(cmd::Seek{.target_us = 5'000'000});
  Refusal lost{RefusalCode::SeekFailed, "lecture : Input/output error"};
  lost.recoverable = true;
  rig.session.handle(StageRefusal{.stage = "source", .refusal = lost});
  CHECK(rig.count("seek 5000000") == 1);
  rig.screen.now_us += 1'500'000;
  rig.tick();
  CHECK(rig.count("seek 5000000") == 1);  // pas encore
  rig.screen.now_us += 600'000;
  rig.tick();
  CHECK(rig.count("seek 5000000") == 2);
  CHECK(rig.stages.log.back() == "seek 5000000");
  // Encore perdu : le délai double (4 s).
  rig.session.handle(StageRefusal{.stage = "source", .refusal = lost});
  rig.screen.now_us += 3'000'000;
  rig.tick();
  CHECK(rig.count("seek 5000000") == 2);
  rig.screen.now_us += 1'100'000;
  rig.tick();
  CHECK(rig.count("seek 5000000") == 3);
  // Le partage revient : l'image arrive, plus rien n'est rejoué.
  rig.session.handle(ring_frame(1, 5'000'000, rig.session.epoch()));
  rig.tick();
  CHECK(rig.shown_pts() == 5'000'000);
  rig.screen.now_us += 20'000'000;
  rig.tick();
  CHECK(rig.count("seek 5000000") == 3);
}

TEST_CASE("Un seek refusé dont la source retrouve sa position n'est pas rejoué (A13-08)", "[app][session]") {
  ajp::app::play::FrenchUi french;
  Rig rig(SessionConfig{.duration_us = 60'000'000});
  rig.session.handle(ring_frame(0, 1'000'000));
  rig.tick();
  rig.session.command(cmd::Seek{.target_us = 5'000'000});
  Refusal lost{RefusalCode::SeekFailed, "lecture"};
  lost.recoverable = true;
  rig.session.handle(StageRefusal{.stage = "source", .refusal = lost});
  rig.session.handle(ring_frame(1, 1'000'000, rig.session.epoch()));  // revenue à l'image tenue
  rig.screen.now_us += 5'000'000;
  rig.tick();
  CHECK(rig.count("seek 5000000") == 1);
}

}  // namespace ajp::app::play
