// Les 9 profils livrés, leur chaîne dans la tranche verticale, et user.toml par-dessus.
#include "app/play/profiles.h"

#include <filesystem>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/profile.h"

namespace ajp::app::play {
namespace {

const core::SourceInfo k1080p{.width = 1920, .height = 1080, .fps = 23.976};

TEST_CASE("La base livre 9 profils lisibles, le 1 au démarrage", "[app][profiles]") {
  auto base = core::parse_profile(base_profiles_toml());
  REQUIRE(base);
  CHECK(base->start_profile == 1);
  for (const auto& profile : base->profiles) {
    REQUIRE(profile);
    CHECK_FALSE(profile->name.empty());
  }
  CHECK(base->profiles[0]->name == "Qualité 1440p");
  CHECK(base->profiles[2]->name == "4K");
}

TEST_CASE("Un profil fixe la chaîne, le reste des réglages ne bouge pas", "[app][profiles]") {
  auto base = core::parse_profile(base_profiles_toml());
  REQUIRE(base);
  PlaySettings mine;
  mine.true_hdr_contrast = 150;
  mine.subtitle_scale = 1.3F;
  std::vector<std::string> notes;
  const PlaySettings quality = apply_profile(*base->profiles[0], k1080p, mine, notes);
  CHECK(quality.work_height == 720);
  CHECK(quality.downscale == Downscale::Area);
  CHECK(quality.rife_factor == 3);
  CHECK(quality.upscale);
  CHECK(quality.model == "2x_AnimeJaNai_HD_V3.1Sharp1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx");
  CHECK(quality.true_hdr);
  CHECK(quality.true_hdr_contrast == 150);
  CHECK(quality.subtitle_scale == 1.3F);
  CHECK(notes.empty());

  const PlaySettings four_k = apply_profile(*base->profiles[2], k1080p, mine, notes);
  CHECK(four_k.work_height == 0);
  CHECK(four_k.rife_factor == 2);
  const PlaySettings source = apply_profile(*base->profiles[8], k1080p, mine, notes);
  CHECK_FALSE(source.upscale);
  CHECK(source.rife_factor == 0);
  CHECK_FALSE(source.true_hdr);
}

TEST_CASE("Les réglages en cours se relisent comme un profil", "[app][profiles]") {
  PlaySettings s;
  s.work_height = 540;
  s.downscale = Downscale::CatmullRom;
  s.rife_factor = 4;
  s.model = "";
  s.true_hdr = true;
  const std::string usual = std::string(kSpanModelId) + ".onnx";
  const core::Profile profile{.name = "Mien", .hdr = hdr_of(s), .rules = rules_of(s, usual)};
  std::vector<std::string> notes;
  const PlaySettings back = apply_profile(profile, k1080p, PlaySettings{}, notes);
  CHECK(back.work_height == 540);
  CHECK(back.downscale == Downscale::CatmullRom);
  CHECK(back.rife_factor == 4);
  CHECK(back.model == usual);
  CHECK(back.true_hdr);
  CHECK(profile_summary(profile, k1080p) == "540p Catmull-Rom → RIFE ×4 → SPAN 1080p · RTX HDR");
  // Un autre modèle est nommé dans le résumé.
  s.model = "2x_AnimeJaNai_HD_V3.1Sharp1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx";
  const core::Profile other{.name = "Net", .hdr = hdr_of(s), .rules = rules_of(s, usual)};
  CHECK(profile_summary(other, k1080p).find("SPAN 1080p (") != std::string::npos);
}

TEST_CASE("Ce que la chaîne ne sait pas faire est dit, pas inventé", "[app][profiles]") {
  core::Profile odd{.hdr = core::HdrMode::Off};
  core::Rule rule{.max_pixels = core::limits::kPixels.hi, .max_fps = core::limits::kFps.hi};
  rule.steps.push_back(
      core::Step{.kind = core::StepKind::Upscale, .model = "m", .factor = {.num = 2, .den = 1}});
  rule.steps.push_back(core::Step{.kind = core::StepKind::Interpolate,
                                  .model = "rife_v4.26",
                                  .factor = {.num = 5, .den = 2},
                                  .policy = core::Policy::Fast});
  odd.rules.push_back(rule);
  std::vector<std::string> notes;
  const PlaySettings s = apply_profile(odd, k1080p, PlaySettings{}, notes);
  CHECK(s.rife_factor == 0);
  REQUIRE(notes.size() == 1);
  CHECK(notes[0].find("5/2") != std::string::npos);
}

TEST_CASE("user.toml : absent, la base ; une modification n'écrit que ce qui diffère", "[app][profiles]") {
  const auto dir = std::filesystem::temp_directory_path() / "ajp-profiles-test";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const auto user = dir / "user.toml";
  auto profiles = load_profiles(user);
  REQUIRE(profiles);
  CHECK(profiles->user_text.empty());
  CHECK(profiles->file.start_profile == 1);

  const std::vector<core::UserEdit> edits{core::edit::StartProfile{.slot = 3},
                                          core::edit::ProfileName{.slot = 2, .name = "Mon ×4"}};
  auto edited = edit_profiles(user, *profiles, edits);
  REQUIRE(edited);
  CHECK(edited->file.start_profile == 3);
  CHECK(edited->file.profiles[1]->name == "Mon ×4");
  auto reread = load_profiles(user);
  REQUIRE(reread);
  CHECK(reread->file == edited->file);
  CHECK(reread->user_text.find("Qualité 1440p") == std::string::npos);  // la base n'est pas recopiée

  const std::vector<core::UserEdit> reset{core::edit::ResetProfile{.slot = 2}};
  auto back = edit_profiles(user, *reread, reset);
  REQUIRE(back);
  CHECK(back->file.profiles[1]->name == "Fluide 1440p ×4");
  std::filesystem::remove_all(dir);
}

}  // namespace
}  // namespace ajp::app::play
