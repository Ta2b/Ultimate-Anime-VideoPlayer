#include "core/profile.h"

#include <array>
#include <string>

#include <catch2/catch_test_macros.hpp>

using ajp::RefusalCode;
using namespace ajp::core;

namespace {

constexpr const char* kValid = R"(
version = 5
start_profile = 2

[[profile]]
slot = 2
name = "Qualité"
hdr = "native"

[[profile.rule]]
max_pixels = 2073600
max_fps = 30
steps = [
  { kind = "downscale", filter = "spline36", height = 720 },
  { kind = "interpolate", model = "rife-4.26", factor = 2, policy = "quality" },
  { kind = "upscale", model = "2x_AnimeJaNai_HD_V3_Compact", factor = 2 },
  { kind = "interpolate", model = "rife-4.26", factor = "5/2", policy = "fast", pacing = "smooth" },
]

[[profile.rule]]
min_pixels = 2073601
steps = []
)";

RefusalCode refusal_of(const std::string& text) {
  auto result = parse_profile(text);
  REQUIRE_FALSE(result);
  return result.error().code;
}

std::string with_steps(const std::string& steps) {
  return "version = 5\n[[profile]]\nslot = 1\n[[profile.rule]]\nsteps = [" + steps + "]\n";
}

}  // namespace

TEST_CASE("Profil : un fichier v5 complet se lit champ par champ", "[profile]") {
  auto file = parse_profile(kValid);
  REQUIRE(file);
  CHECK(file->start_profile == 2);
  CHECK_FALSE(file->profiles[0]);
  REQUIRE(file->profiles[1]);

  const Profile& profile = *file->profiles[1];
  CHECK(profile.name == "Qualité");
  CHECK(profile.hdr == HdrMode::Native);
  REQUIRE(profile.rules.size() == 2);

  const Rule& rule = profile.rules[0];
  CHECK(rule.min_pixels == 0);
  CHECK(rule.max_pixels == 2073600);
  CHECK(rule.max_fps == 30.0);
  REQUIRE(rule.steps.size() == 4);
  CHECK(rule.steps[0].kind == StepKind::Downscale);
  CHECK(rule.steps[0].filter == DownscaleFilter::Spline36);
  CHECK(rule.steps[0].height == 720);
  CHECK(rule.steps[1].factor == Ratio{2, 1});
  CHECK(rule.steps[2].model == "2x_AnimeJaNai_HD_V3_Compact");
  CHECK(rule.steps[3].factor == Ratio{5, 2});
  CHECK(rule.steps[3].policy == Policy::Fast);
  CHECK(rule.steps[3].pacing == Pacing::Smooth);

  CHECK(profile.rules[1].min_pixels == 2073601);
  CHECK(profile.rules[1].max_pixels == limits::kPixels.hi);
}

TEST_CASE("Profil : les erreurs de forme sont des refus nommés", "[profile]") {
  CHECK(refusal_of("version = ") == RefusalCode::ProfileUnreadable);
  CHECK(refusal_of("start_profile = 1") == RefusalCode::FieldMissing);
  CHECK(refusal_of("version = 4") == RefusalCode::ProfileVersion);
  CHECK(refusal_of("version = \"5\"") == RefusalCode::FieldType);
  CHECK(refusal_of(with_steps(R"({ kind = "upscale", factor = 2 })")) == RefusalCode::FieldMissing);
  CHECK(refusal_of(with_steps(R"({ kind = "teleport", model = "x", factor = 2 })")) ==
        RefusalCode::FieldOutOfRange);
  CHECK(refusal_of(with_steps(R"({ kind = "interpolate", model = "r", factor = "2:1", policy = "fast" })")) ==
        RefusalCode::FieldType);
}

TEST_CASE("Profil : les filtres de réduction du lecteur se lisent par leur nom", "[profile]") {
  for (const auto& [text, filter] : std::array<std::pair<const char*, DownscaleFilter>, 6>{{
           {"area", DownscaleFilter::Area},
           {"spline36", DownscaleFilter::Spline36},
           {"bilinear", DownscaleFilter::Bilinear},
           {"bicubic", DownscaleFilter::Bicubic},
           {"mitchell", DownscaleFilter::Mitchell},
           {"catmull-rom", DownscaleFilter::CatmullRom},
       }}) {
    auto file = parse_profile(
        with_steps(std::string(R"({ kind = "downscale", filter = ")") + text + R"(", height = 720 })"));
    REQUIRE(file);
    CHECK(file->profiles[0]->rules[0].steps[0].filter == filter);
    // Et s'écrit à l'identique.
    const auto again = parse_profile(format_profile(*file));
    REQUIRE(again);
    CHECK(*again == *file);
  }
  CHECK(refusal_of(with_steps(R"({ kind = "downscale", filter = "lanczos", height = 720 })")) ==
        RefusalCode::FieldOutOfRange);
}

TEST_CASE("Profil : une valeur hors plage est refusée, jamais corrigée", "[profile]") {
  CHECK(refusal_of("version = 5\nstart_profile = 10") == RefusalCode::FieldOutOfRange);
  CHECK(refusal_of("version = 5\nstart_profile = 3\n[[profile]]\nslot = 1") == RefusalCode::FieldOutOfRange);
  CHECK(refusal_of(with_steps(R"({ kind = "upscale", model = "m", factor = 9 })")) ==
        RefusalCode::FieldOutOfRange);
  CHECK(refusal_of(with_steps(R"({ kind = "upscale", model = "m", factor = "3/2" })")) ==
        RefusalCode::FieldType);
  CHECK(refusal_of(with_steps(R"({ kind = "interpolate", model = "r", factor = 1, policy = "fast" })")) ==
        RefusalCode::FieldOutOfRange);
  CHECK(refusal_of(with_steps(R"({ kind = "downscale", filter = "area", height = 100 })")) ==
        RefusalCode::FieldOutOfRange);
  CHECK(refusal_of("version = 5\n[[profile]]\nslot = 1\n[[profile]]\nslot = 1") ==
        RefusalCode::FieldOutOfRange);
}

TEST_CASE("Profil : au plus 16 étapes, au produit de facteurs au plus 16", "[profile]") {
  std::string seventeen;
  for (int i = 0; i < 17; ++i) seventeen += R"({ kind = "downscale", filter = "area", height = 720 },)";
  CHECK(refusal_of(with_steps(seventeen)) == RefusalCode::TooManySteps);

  const std::string x4 = R"({ kind = "upscale", model = "m", factor = 4 },)";
  const std::string x8 = R"({ kind = "interpolate", model = "r", factor = 8, policy = "fast" })";
  CHECK(refusal_of(with_steps(x4 + x8)) == RefusalCode::FactorProductTooLarge);
  CHECK(parse_profile(
      with_steps(x4 + R"({ kind = "interpolate", model = "r", factor = 4, policy = "fast" })")));
}
