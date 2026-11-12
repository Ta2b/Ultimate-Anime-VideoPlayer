// C09-01 : un refus d'un réseau pendant la lecture nomme l'étage qui l'a rendu ; un refus qui ne vient
// d'aucun réseau (ingestion) n'en nomme aucun. Le repli (stage_fallback.h) ne retire que l'étage nommé.
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/links.h"
#include "app/play/upscale_stage.h"
#include "stage_rig.h"
#include "upscale_rig.h"

namespace ajp::app::play {
namespace {

using namespace std::chrono_literals;
using upscale_test::Rig;

std::vector<Refusal> refused_by(const std::string& engine, bool with_rife) {
  Rig rig;
  rig.backend.refuse_engine(engine);
  rig.start(rig.setup(with_rife, Factor{.num = 2, .den = 1}));
  rig.feed(1, 4);
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (rig.reported().empty() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(1ms);
  rig.stage.reset();
  return rig.reported();
}

}  // namespace

TEST_CASE("C09-01 : un refus de SPAN en lecture nomme SPAN", "[app][stage]") {
  const std::vector<Refusal> reported = refused_by("span", false);
  REQUIRE_FALSE(reported.empty());
  CHECK(is_fatal_stage_key(reported[0].key));
  CHECK(blamed_stage_key(reported[0].key) == kSpanRefusalKey);
  CHECK(reported[0].code == RefusalCode::VendorCallFailed);  // le code reste (A31-01)
}

TEST_CASE("C09-01 : un refus de RIFE en lecture nomme l'interpolation", "[app][stage]") {
  const std::vector<Refusal> reported = refused_by("rife", true);
  REQUIRE_FALSE(reported.empty());
  CHECK(is_fatal_stage_key(reported[0].key));
  CHECK(blamed_stage_key(reported[0].key) == kMotionRefusalKey);
}

TEST_CASE("C09-01 : les clés des refus d'étage", "[app][stage]") {
  CHECK(fatal_stage_key("") == kFatalStageKey);
  CHECK(fatal_stage_key("audio.delay_ms") == kFatalStageKey);
  CHECK(fatal_stage_key(kSpanRefusalKey) == "stage.fatal|stage.span");
  CHECK(is_fatal_stage_key("stage.fatal|stage.motion"));
  CHECK_FALSE(is_fatal_stage_key("stage.fatalx"));
  CHECK_FALSE(is_fatal_stage_key(kSpanRefusalKey));
  CHECK(blamed_stage_key(kFatalStageKey).empty());
  CHECK(blamed_stage_key(kRingRefusalKey) == kRingRefusalKey);
  CHECK(blamed_stage_key("autre").empty());
  // Une clé déjà posée reste.
  std::expected<void, Refusal> keyed = std::unexpected(Refusal{.code = RefusalCode::Cancelled, .key = "x"});
  CHECK(blame_stage(keyed, kSpanRefusalKey).error().key == "x");
  CHECK(blame_stage({}, kSpanRefusalKey).has_value());
}

}  // namespace ajp::app::play
