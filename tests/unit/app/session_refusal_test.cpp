// La session et les refus des threads : fatal ou non, échec ou fin d'épisode, réouverture sur un
// changement de taille (découpé de session_test.cpp, règle des 800 lignes).
#include <utility>

#include <catch2/catch_test_macros.hpp>

#include "app/play/session.h"
#include "doubles.h"
#include "french_ui.h"

namespace ajp::app::play {
namespace {

using test::FakeClock;
using test::FakeScreen;
using test::FakeStages;

struct Rig {
  FakeStages stages;
  FakeScreen screen;
  FakeClock clock;
  Session session;

  explicit Rig(SessionConfig config = {}) : session(std::move(config), stages, screen, clock) {}
};

TEST_CASE("Un refus fatal d'un thread arrête la lecture et s'affiche", "[app][session]") {
  ajp::app::play::FrenchUi french;
  Rig rig;
  rig.session.handle(
      StageRefusal{.stage = "gpu", .refusal = {RefusalCode::ThreadFailed, "boum"}, .fatal = true});
  CHECK(rig.session.finished());
  REQUIRE(rig.session.refusals().size() == 1);
  CHECK(rig.session.refusals()[0] == "gpu : ThreadFailed boum");
}

TEST_CASE("Un refus fatal est un échec, pas une fin d'épisode ; un refus non fatal ne l'est pas",
          "[app][session]") {
  Rig rig;
  rig.session.handle(StageRefusal{.stage = "audio", .refusal = {RefusalCode::AudioDecodeFailed, "paquet"}});
  CHECK_FALSE(rig.session.failure().has_value());
  CHECK_FALSE(rig.session.finished());
  rig.session.handle(
      StageRefusal{.stage = "source", .refusal = {RefusalCode::DecodeFailed, "lecture"}, .fatal = true});
  rig.session.handle(
      StageRefusal{.stage = "gpu", .refusal = {RefusalCode::ThreadFailed, "boum"}, .fatal = true});
  CHECK(rig.session.finished());
  REQUIRE(rig.session.failure().has_value());
  CHECK(rig.session.failure()->code == RefusalCode::DecodeFailed);  // le premier, la cause
  CHECK(rig.session.failure()->detail == "source : lecture");
}

TEST_CASE("Un changement de taille de la source arrête la session et demande la réouverture",
          "[app][session]") {
  ajp::app::play::FrenchUi french;
  Rig rig;
  CHECK_FALSE(rig.session.resize_asked());
  rig.session.handle(StageRefusal{.stage = "gpu", .refusal = {RefusalCode::SourceResized, "1920x1080"}});
  CHECK(rig.session.resize_asked());
  CHECK(rig.session.finished());
  CHECK_FALSE(rig.session.failure().has_value());  // une réouverture, pas un échec
}

// A13-02 : la source abandonne la lecture (NAS débranché) par un refus recoverable, puis rend la fin
// du flux. Ce n'est pas la fin de la vidéo : un échec, qui n'enchaîne pas l'épisode suivant.
TEST_CASE("Une fin de flux après abandon de lecture est un échec, pas une fin d'épisode", "[app][session]") {
  Rig rig;
  Refusal lost{RefusalCode::DecodeFailed, "lecture d'un paquet, après 6 essais"};
  lost.recoverable = true;
  rig.session.handle(StageRefusal{.stage = "source", .refusal = lost});
  CHECK_FALSE(rig.session.failure().has_value());  // les images déjà décodées passent encore
  rig.session.handle(StreamEnded{.epoch = rig.session.epoch()});
  REQUIRE(rig.session.failure().has_value());
  CHECK(rig.session.failure()->code == RefusalCode::DecodeFailed);
  CHECK(rig.session.failure()->detail == "source : lecture d'un paquet, après 6 essais");
}

TEST_CASE("Une vraie fin de flux reste une fin d'épisode", "[app][session]") {
  Rig rig;
  rig.session.handle(StageRefusal{.stage = "audio", .refusal = {RefusalCode::AudioDecodeFailed, "paquet"}});
  rig.session.handle(StreamEnded{.epoch = rig.session.epoch()});
  CHECK_FALSE(rig.session.failure().has_value());
}

}  // namespace
}  // namespace ajp::app::play
