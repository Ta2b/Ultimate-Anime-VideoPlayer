#include "core/resolve.h"

#include <cmath>
#include <limits>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using ajp::RefusalCode;
using Catch::Approx;
using namespace ajp::core;

namespace {

const Capabilities kCaps{.models = {"span-2x", "rife"}};
const SourceInfo k1080p24{.width = 1920, .height = 1080, .fps = 23.976};
const DisplayInfo kSdrDisplay{.width = 2560, .height = 1440, .refresh_hz = 165.0, .hdr = false};

Step upscale(int factor) {
  return Step{.kind = StepKind::Upscale, .model = "span-2x", .factor = {factor, 1}};
}

Step interpolate(Policy policy, Pacing pacing = Pacing::Source, Ratio factor = {2, 1}) {
  return Step{
      .kind = StepKind::Interpolate, .model = "rife", .factor = factor, .policy = policy, .pacing = pacing};
}

Profile with_steps(std::vector<Step> steps, HdrMode hdr = HdrMode::Off) {
  return Profile{
      .name = "test",
      .hdr = hdr,
      .rules = {Rule{.max_pixels = limits::kPixels.hi, .max_fps = 1000.0, .steps = std::move(steps)}}};
}

}  // namespace

TEST_CASE("resolve : sans règle, la vidéo passe telle quelle", "[resolve]") {
  Profile profile{.rules = {Rule{.min_pixels = 0, .max_pixels = 1280 * 720, .max_fps = 1000.0}}};
  auto plan = resolve(profile, kCaps, k1080p24, kSdrDisplay);
  REQUIRE(plan);
  CHECK_FALSE(plan->upscale);
  CHECK(plan->levels.empty());
  CHECK(plan->output_width == 1920);
  CHECK(plan->output_fps == 23.976);
}

TEST_CASE("resolve : la première règle qui correspond gagne, bornes incluses", "[resolve]") {
  Profile profile{
      .rules = {
          Rule{
              .min_pixels = 1920 * 1080, .max_pixels = 1920 * 1080, .max_fps = 23.976, .steps = {upscale(2)}},
          Rule{.max_pixels = limits::kPixels.hi, .max_fps = 1000.0, .steps = {upscale(4)}},
      }};
  const Rule* rule = match_rule(profile, k1080p24);
  REQUIRE(rule == &profile.rules[0]);
  CHECK(match_rule(profile, {.width = 1920, .height = 1080, .fps = 30.0}) == &profile.rules[1]);
}

TEST_CASE("resolve : les ordonnancements S1, S2 et S3", "[resolve]") {
  SECTION("Quality avant l'agrandissement donne S1") {
    auto plan = resolve(with_steps({interpolate(Policy::Quality), upscale(2)}), kCaps, k1080p24, kSdrDisplay);
    REQUIRE(plan);
    REQUIRE(plan->levels.size() == 1);
    CHECK(plan->levels[0].schedule == Schedule::BeforeUpscale);
    CHECK(plan->output_width == 3840);
    // 165 Hz fixe : x2 devient la fraction calée sur 3 vsyncs, 55 im/s (AH-375).
    REQUIRE(plan->cadence);
    CHECK(plan->cadence->grid == Grid::Vsync);
    CHECK(plan->cadence->vsyncs_per_frame == 3);
    CHECK(plan->output_fps == Approx(55.0));
    CHECK(plan->levels[0].factor == plan->cadence->factor);
  }
  SECTION("Quality après l'agrandissement donne S2") {
    auto plan = resolve(with_steps({upscale(2), interpolate(Policy::Quality)}), kCaps, k1080p24, kSdrDisplay);
    REQUIRE(plan);
    CHECK(plan->levels[0].schedule == Schedule::AfterUpscale);
  }
  SECTION("Quality x2 puis Fast x2 donne S1 puis S3") {
    auto plan = resolve(with_steps({interpolate(Policy::Quality), interpolate(Policy::Fast), upscale(2)}),
                        kCaps, k1080p24, kSdrDisplay);
    REQUIRE(plan);
    REQUIRE(plan->levels.size() == 2);
    CHECK(plan->levels[0].schedule == Schedule::BeforeUpscale);
    CHECK(plan->levels[1].schedule == Schedule::OutputRes);
    CHECK(plan->output_fps == 23.976 * 4);
  }
}

TEST_CASE("resolve : la réduction ne s'applique qu'à une source plus haute", "[resolve]") {
  const Step down{.kind = StepKind::Downscale, .filter = DownscaleFilter::Spline36, .height = 720};
  auto plan = resolve(with_steps({down, upscale(2)}), kCaps, k1080p24, kSdrDisplay);
  REQUIRE(plan);
  REQUIRE(plan->downscale);
  CHECK(plan->output_width == 2560);
  CHECK(plan->output_height == 1440);

  auto small = resolve(with_steps({down, upscale(2)}), kCaps, {.width = 1280, .height = 720, .fps = 24.0},
                       kSdrDisplay);
  REQUIRE(small);
  CHECK_FALSE(small->downscale);
  CHECK(small->output_height == 1440);
}

TEST_CASE("resolve : les ordres sans ordonnancement connu sont refusés", "[resolve]") {
  const Step down{.kind = StepKind::Downscale, .filter = DownscaleFilter::Area, .height = 720};
  auto late_down = resolve(with_steps({upscale(2), down}), kCaps, k1080p24, kSdrDisplay);
  REQUIRE_FALSE(late_down);
  CHECK(late_down.error().code == RefusalCode::StepOrder);

  auto split =
      resolve(with_steps({upscale(2), interpolate(Policy::Fast), upscale(2)}), kCaps, k1080p24, kSdrDisplay);
  REQUIRE_FALSE(split);
  CHECK(split.error().code == RefusalCode::StepOrder);
}

TEST_CASE("resolve : Quality impose le rythme Source", "[resolve]") {
  auto plan =
      resolve(with_steps({interpolate(Policy::Quality, Pacing::Smooth)}), kCaps, k1080p24, kSdrDisplay);
  REQUIRE_FALSE(plan);
  CHECK(plan.error().code == RefusalCode::PacingForbidden);
  CHECK(resolve(with_steps({interpolate(Policy::Fast, Pacing::KeepPoses)}), kCaps, k1080p24, kSdrDisplay));
}

TEST_CASE("resolve : un modèle absent ou une route HDR impossible sont refusés", "[resolve]") {
  Step unknown = upscale(2);
  unknown.model = "absent";
  auto missing = resolve(with_steps({unknown}), kCaps, k1080p24, kSdrDisplay);
  REQUIRE_FALSE(missing);
  CHECK(missing.error().code == RefusalCode::ModelUnavailable);

  auto sdr = resolve(with_steps({upscale(2)}, HdrMode::Native), kCaps, k1080p24, kSdrDisplay);
  REQUIRE_FALSE(sdr);
  CHECK(sdr.error().code == RefusalCode::HdrUnavailable);

  DisplayInfo hdr_display = kSdrDisplay;
  hdr_display.hdr = true;
  auto no_sdk = resolve(with_steps({upscale(2)}, HdrMode::TrueHdr), kCaps, k1080p24, hdr_display);
  REQUIRE_FALSE(no_sdk);
  CHECK(no_sdk.error().code == RefusalCode::HdrUnavailable);

  auto native = resolve(with_steps({upscale(2)}, HdrMode::Native), kCaps, k1080p24, hdr_display);
  REQUIRE(native);
  CHECK(native->hdr.kind == HdrMode::Native);
}

TEST_CASE("resolve : le hachage suit le contenu du Plan", "[resolve]") {
  auto a = resolve(with_steps({upscale(2)}), kCaps, k1080p24, kSdrDisplay);
  auto b = resolve(with_steps({upscale(2)}), kCaps, k1080p24, kSdrDisplay);
  auto c = resolve(with_steps({upscale(2), interpolate(Policy::Fast)}), kCaps, k1080p24, kSdrDisplay);
  REQUIRE((a && b && c));
  CHECK(a->hash == b->hash);
  CHECK(a->hash != c->hash);
  CHECK(a->hash == compute_hash(*a));
}

TEST_CASE("resolve : sur un écran fixe, l'interpolation unique prend le facteur calé", "[resolve][pacing]") {
  const SourceInfo ntsc{.width = 1920, .height = 1080, .fps = 24000.0 / 1001.0};
  const DisplayInfo at_120{.width = 3840, .height = 2160, .refresh_hz = 120.0};

  auto x4 = resolve(with_steps({interpolate(Policy::Fast, Pacing::Source, {4, 1})}), kCaps, ntsc, at_120);
  REQUIRE(x4);
  REQUIRE(x4->levels.size() == 1);
  CHECK(x4->levels[0].factor == Ratio{5, 1});
  REQUIRE(x4->cadence);
  CHECK(x4->cadence->grid == Grid::Vsync);
  CHECK(x4->cadence->vsyncs_per_frame == 1);
  CHECK(x4->cadence->speed == Approx(1.001));
  CHECK(x4->output_fps == Approx(ntsc.fps * 5.0));

  auto x2 = resolve(with_steps({interpolate(Policy::Fast)}), kCaps, ntsc, at_120);
  REQUIRE(x2);
  CHECK(x2->levels[0].factor == Ratio{5, 3});
  CHECK(x2->cadence->vsyncs_per_frame == 3);

  // Deux interpolations : les facteurs restent, la cadence de leur produit est seulement calée.
  auto two =
      resolve(with_steps({interpolate(Policy::Quality), interpolate(Policy::Fast)}), kCaps, ntsc, at_120);
  REQUIRE(two);
  CHECK(two->levels[0].factor == Ratio{2, 1});
  CHECK(two->levels[1].factor == Ratio{2, 1});
  CHECK(two->cadence->factor == Ratio{4, 1});
}

TEST_CASE("resolve : VRR garde le facteur, un écran inconnu ne cale rien", "[resolve][pacing]") {
  const Profile profile = with_steps({interpolate(Policy::Fast, Pacing::Source, {4, 1})});
  auto vrr =
      resolve(profile, kCaps, k1080p24, {.width = 2560, .height = 1440, .refresh_hz = 165.0, .vrr = true});
  REQUIRE(vrr);
  CHECK(vrr->levels[0].factor == Ratio{4, 1});
  REQUIRE(vrr->cadence);
  CHECK(vrr->cadence->grid == Grid::Timestamps);
  CHECK(vrr->output_fps == Approx(23.976 * 4));

  auto unknown = resolve(profile, kCaps, k1080p24, DisplayInfo{});
  REQUIRE(unknown);
  CHECK_FALSE(unknown->cadence);
  CHECK(unknown->levels[0].factor == Ratio{4, 1});
  CHECK(unknown->hash != vrr->hash);

  // Sans règle, rien à caler, même sur un écran connu.
  Profile none{.rules = {Rule{.max_pixels = 1280 * 720, .max_fps = 1000.0}}};
  auto passthrough = resolve(none, kCaps, k1080p24, kSdrDisplay);
  REQUIRE(passthrough);
  CHECK_FALSE(passthrough->cadence);
}

TEST_CASE("resolve : un écran hors bornes est refusé", "[resolve][pacing]") {
  const Profile profile = with_steps({interpolate(Policy::Fast)});
  const auto code_of = [&](const DisplayInfo& display) {
    auto plan = resolve(profile, kCaps, k1080p24, display);
    REQUIRE_FALSE(plan);
    return plan.error().code;
  };
  constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
  CHECK(code_of({.refresh_hz = -60.0}) == RefusalCode::FieldOutOfRange);
  CHECK(code_of({.refresh_hz = 2000.0}) == RefusalCode::FieldOutOfRange);
  CHECK(code_of({.refresh_hz = kNan}) == RefusalCode::FieldOutOfRange);
  DisplayInfo dim = kSdrDisplay;
  dim.peak_nits = kNan;
  CHECK(code_of(dim) == RefusalCode::FieldOutOfRange);
  dim.peak_nits = 400.0;
  dim.min_nits = 400.0;
  CHECK(code_of(dim) == RefusalCode::FieldOutOfRange);
  dim.min_nits = 0.05;
  dim.sdr_white_nits = 0.0;
  CHECK(code_of(dim) == RefusalCode::FieldOutOfRange);
  dim.sdr_white_nits = 240.0;
  CHECK(resolve(profile, kCaps, k1080p24, dim));
}

TEST_CASE("resolve : TrueHDR bloqué rend le texte du premier bloqueur", "[resolve]") {
  DisplayInfo hdr_display = kSdrDisplay;
  hdr_display.hdr = true;
  const Profile profile = with_steps({upscale(2)}, HdrMode::TrueHdr);

  auto unchecked = resolve(profile, kCaps, k1080p24, hdr_display);
  REQUIRE_FALSE(unchecked);
  CHECK(unchecked.error().code == RefusalCode::HdrUnavailable);
  CHECK(unchecked.error().detail == kTrueHdrUnchecked);

  Capabilities blocked = kCaps;
  blocked.true_hdr_blocker = "le GPU de rendu n'est pas un GPU NVIDIA";
  auto refused = resolve(profile, blocked, k1080p24, hdr_display);
  REQUIRE_FALSE(refused);
  CHECK(refused.error().detail == blocked.true_hdr_blocker);

  Capabilities ready = kCaps;
  ready.true_hdr_blocker.clear();
  auto plan = resolve(profile, ready, k1080p24, hdr_display);
  REQUIRE(plan);
  CHECK(plan->hdr.kind == HdrMode::TrueHdr);
}
