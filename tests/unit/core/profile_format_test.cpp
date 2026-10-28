// format_profile : l'inverse exact de parse_profile.
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "core/profile.h"

using namespace ajp::core;

namespace {

ProfileFile sample() {
  ProfileFile file{.start_profile = 3};
  file.profiles[2] = Profile{
      .name = "Qualité \"maison\" \\ ☆\ttab",
      .hdr = HdrMode::TrueHdr,
      .rules = {
          Rule{
              .min_pixels = 921'601,
              .max_pixels = 2'073'600,
              .min_fps = 23.976,
              .max_fps = 30.0,
              .steps = {Step{.kind = StepKind::Downscale, .filter = DownscaleFilter::Spline36, .height = 720},
                        Step{.kind = StepKind::Interpolate,
                             .model = "rife-4.26",
                             .factor = {5, 2},
                             .policy = Policy::Fast,
                             .pacing = Pacing::KeepPoses},
                        Step{.kind = StepKind::Upscale, .model = "2x_HD", .factor = {2, 1}}}},
          Rule{.max_pixels = limits::kPixels.hi, .max_fps = limits::kFps.hi}}};
  file.profiles[8] = Profile{.name = ""};
  return file;
}

}  // namespace

TEST_CASE("Profil : format_profile se relit à l'identique", "[profile]") {
  const ProfileFile file = sample();
  const std::string text = format_profile(file);
  auto reread = parse_profile(text);
  REQUIRE(reread);
  CHECK(*reread == file);
  CHECK(format_profile(*reread) == text);
}

TEST_CASE("Profil : le texte écrit omet les valeurs par défaut des bornes et du rythme", "[profile]") {
  const std::string text = format_profile(sample());
  CHECK(text.starts_with("version = 5\nstart_profile = 3\n"));
  CHECK(text.find("slot = 3\nname = \"Qualité \\\"maison\\\" \\\\ ☆\\u0009tab\"\nhdr = \"truehdr\"") !=
        std::string::npos);
  CHECK(text.find("min_fps = 23.976\nmax_fps = 30.0\n") != std::string::npos);
  CHECK(text.find("factor = \"5/2\", policy = \"fast\", pacing = \"keep_poses\"") != std::string::npos);
  CHECK(text.find("{ kind = \"upscale\", model = \"2x_HD\", factor = 2 }") != std::string::npos);
  CHECK(text.find("[[profile.rule]]\nsteps = []\n") != std::string::npos);
  CHECK(text.find("slot = 9\nname = \"\"\nhdr = \"off\"\n") != std::string::npos);
  CHECK(format_profile(ProfileFile{}) == "version = 5\nstart_profile = 1\n");
}
