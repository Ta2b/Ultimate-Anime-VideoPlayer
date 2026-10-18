#include "core/ini_import.h"

#include <algorithm>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>

#include <catch2/catch_test_macros.hpp>

using ajp::RefusalCode;
using namespace ajp::core;
namespace fs = std::filesystem;

namespace {

// Fichier type écrit par le gestionnaire avant les étapes ordonnées (format 1).
constexpr const char* kLegacy = R"([global]
logging=yes
backend=TensorRT
default_slot=2

[slot_2]
profile_name=Mon profil
chain_1_min_resolution=1280x720
chain_1_max_resolution=1920x1080
chain_1_min_fps=0
chain_1_max_fps=31
chain_1_model_1_name=2x_AnimeJaNai_HD_V3.1_Balanced
chain_1_model_1_resize_height_before_upscale=0
chain_1_model_1_resize_factor_before_upscale=100
chain_1_rife=yes
chain_1_rife_model=426
chain_1_rife_factor_numerator=2
chain_1_rife_factor_denominator=1
chain_1_rife_before_upscale=yes
chain_2_min_resolution=0x0
chain_2_max_resolution=1280x720
chain_2_max_fps=0
chain_2_model_1_name=2x_AnimeJaNai_SD_V1beta34_Compact
chain_2_rife=no
)";

// Format 4 : étapes ordonnées, avec les clés historiques recopiées en miroir.
constexpr const char* kStages = R"([global]
config_version=4
backend=TensorRT
rife_engine=quality
default_slot=1

[slot_1]
profile_name=Ordonné
chain_1_min_resolution=0x0
chain_1_max_resolution=0x0
chain_1_min_fps=0
chain_1_max_fps=0
chain_1_pipeline_version=1
chain_1_stage_count=4
chain_1_stage_1_id=a
chain_1_stage_1_kind=resize
chain_1_stage_1_method=area
chain_1_stage_1_runtime=cuda
chain_1_stage_1_target_height=720
chain_1_stage_2_id=b
chain_1_stage_2_kind=interpolate
chain_1_stage_2_method=rife
chain_1_stage_2_model=4141
chain_1_stage_2_factor_numerator=2
chain_1_stage_2_factor_denominator=1
chain_1_stage_2_policy=quality
chain_1_stage_2_ensemble=yes
chain_1_stage_3_id=c
chain_1_stage_3_kind=upscale
chain_1_stage_3_method=onnx
chain_1_stage_3_model=2x_AnimeJaNai_HD_V3.1_Performance
chain_1_stage_4_id=d
chain_1_stage_4_kind=hdr
chain_1_stage_4_method=nvidia-truehdr
chain_1_stage_4_contrast=110
chain_1_rife=yes
chain_1_rife_model=4141
chain_1_model_1_name=2x_AnimeJaNai_HD_V3.1_Performance
)";

ImportResult imported(const std::string& text) {
  auto result = import_ini(text);
  if (!result) FAIL(result.error().detail);
  return *result;
}

ajp::Refusal refusal_of(const std::string& text) {
  auto result = import_ini(text);
  REQUIRE_FALSE(result);
  return result.error();
}

bool has_note(const ImportResult& result, const std::string& fragment) {
  return std::ranges::any_of(result.notes,
                             [&](const ImportNote& n) { return n.text.find(fragment) != std::string::npos; });
}

// Une chaîne d'un profil de slot 1, avec des clés en plus.
std::string chain(const std::string& keys, const std::string& global = "") {
  return "[global]\n" + global + "default_slot=1\n[slot_1]\nprofile_name=P\n" + keys;
}

std::vector<Step> steps_of(const std::string& text) {
  const ImportResult result = imported(text);
  REQUIRE(result.profile.profiles[0]);
  REQUIRE(result.profile.profiles[0]->rules.size() == 1);
  return result.profile.profiles[0]->rules[0].steps;
}

}  // namespace

TEST_CASE("Import INI : un fichier historique devient un profil v5 équivalent", "[ini_import]") {
  const ImportResult result = imported(kLegacy);
  CHECK(result.source_version == 1);
  CHECK(result.profile.start_profile == 2);
  REQUIRE(result.profile.profiles[1]);
  const Profile& profile = *result.profile.profiles[1];
  CHECK(profile.name == "Mon profil");
  CHECK(profile.hdr == HdrMode::Off);
  REQUIRE(profile.rules.size() == 2);

  const Rule& hd = profile.rules[0];
  CHECK(hd.min_pixels == 1280 * 720);
  CHECK(hd.max_pixels == 1920 * 1080);
  CHECK(hd.min_fps == 0.0);
  CHECK(hd.max_fps == 31.0);
  REQUIRE(hd.steps.size() == 2);
  CHECK(hd.steps[0] == Step{.kind = StepKind::Interpolate,
                            .model = "rife-4.26",
                            .factor = {2, 1},
                            .policy = Policy::Fast,
                            .pacing = Pacing::Smooth});
  CHECK(hd.steps[1] ==
        Step{.kind = StepKind::Upscale, .model = "2x_AnimeJaNai_HD_V3.1_Balanced", .factor = {2, 1}});

  const Rule& sd = profile.rules[1];
  CHECK(sd.min_pixels == 0);
  CHECK(sd.max_pixels == 1280 * 720);
  CHECK(sd.max_fps == limits::kFps.hi);
  REQUIRE(sd.steps.size() == 1);
  CHECK(sd.steps[0].model == "2x_AnimeJaNai_SD_V1beta34_Compact");

  // Le texte v5 se relit en ce même profil, et ce qui disparaît est annoncé avec sa ligne.
  CHECK(parse_profile(result.toml) == result.profile);
  CHECK(has_note(result, "backend"));
  CHECK(has_note(result, "logging"));
  CHECK(std::ranges::is_sorted(result.notes, {}, &ImportNote::line));
  CHECK(result.notes.front().line == 2);
}

TEST_CASE("Import INI : étapes ordonnées du format 4, miroirs ignorés", "[ini_import]") {
  const ImportResult result = imported(kStages);
  CHECK(result.source_version == 4);
  REQUIRE(result.profile.profiles[0]);
  const Profile& profile = *result.profile.profiles[0];
  CHECK(profile.hdr == HdrMode::TrueHdr);
  REQUIRE(profile.rules.size() == 1);
  const Rule& rule = profile.rules[0];
  CHECK(rule.max_pixels == limits::kPixels.hi);
  CHECK(rule.max_fps == limits::kFps.hi);
  REQUIRE(rule.steps.size() == 3);
  CHECK(rule.steps[0] == Step{.kind = StepKind::Downscale, .filter = DownscaleFilter::Area, .height = 720});
  CHECK(rule.steps[1] == Step{.kind = StepKind::Interpolate,
                              .model = "rife-4.14-lite-ensemble",
                              .factor = {2, 1},
                              .policy = Policy::Quality,
                              .pacing = Pacing::Source});
  CHECK(rule.steps[2].model == "2x_AnimeJaNai_HD_V3.1_Performance");
  CHECK(has_note(result, "HDR"));
  CHECK_FALSE(has_note(result, "chain_1_rife"));
  CHECK_FALSE(has_note(result, "chain_1_model_1_name"));
  CHECK(parse_profile(result.toml) == result.profile);
}

TEST_CASE("Import INI : RIFE garde son code, sa politique et son rythme", "[ini_import]") {
  const std::string rife = "chain_1_model_1_name=2x_m\nchain_1_rife=yes\nchain_1_rife_factor_numerator=2\n";
  CHECK(steps_of(chain(rife))[0].model == "rife-4.14");  // code 414 par défaut
  CHECK(steps_of(chain(rife + "chain_1_rife_model=46\n"))[0].model == "rife-4.6");
  CHECK(steps_of(chain(rife + "chain_1_rife_model=4220\n"))[0].model == "rife-4.22");

  const Step quality = steps_of(chain(rife, "rife_engine=old\n"))[0];
  CHECK(quality.policy == Policy::Quality);
  CHECK(quality.pacing == Pacing::Source);
  CHECK(steps_of(chain(rife, "rife_drawings=no\n"))[0].pacing == Pacing::Source);

  const auto after = steps_of(chain(rife + "chain_1_rife_before_upscale=no\n"));
  REQUIRE(after.size() == 2);
  CHECK(after[0].kind == StepKind::Upscale);
  CHECK(after[1].policy == Policy::Fast);
  CHECK(after[1].pacing == Pacing::Source);

  CHECK(steps_of(chain(rife + "chain_1_rife_factor_denominator=4\nchain_1_rife_factor_numerator=10\n"))[0]
            .factor == Ratio{5, 2});
  const ImportResult unit = imported(chain("chain_1_model_1_name=2x_m\nchain_1_rife=yes\n"));
  CHECK(unit.profile.profiles[0]->rules[0].steps.size() == 1);
  CHECK(has_note(unit, "facteur 1"));

  const auto temporal = imported(chain(rife + "chain_1_model_1_frames=5\n"));
  CHECK(temporal.profile.profiles[0]->rules[0].steps[1].kind == StepKind::Interpolate);
  CHECK(has_note(temporal, "temporel"));
}

TEST_CASE("Import INI : la réduction du premier modèle passe en tête", "[ini_import]") {
  const auto steps = steps_of(
      chain("chain_1_model_1_name=4x_m\nchain_1_model_1_resize_height_before_upscale=540\nchain_1_rife=yes\n"
            "chain_1_rife_factor_numerator=2\nchain_1_model_2_name=1x_clean\n"));
  REQUIRE(steps.size() == 4);
  CHECK(steps[0] == Step{.kind = StepKind::Downscale, .filter = DownscaleFilter::Spline36, .height = 540});
  CHECK(steps[1].kind == StepKind::Interpolate);
  CHECK(steps[2] == Step{.kind = StepKind::Upscale, .model = "4x_m", .factor = {4, 1}});
  CHECK(steps[3].factor == Ratio{1, 1});

  const ImportResult directml = imported(chain(
      "chain_1_model_1_name=2x_m\nchain_1_model_1_resize_height_before_upscale=720\n", "backend=Vulkan\n"));
  CHECK(has_note(directml, "bilinéaire"));
}

TEST_CASE("Import INI : ce qui n'a pas d'équivalent v5 est refusé avec sa ligne", "[ini_import]") {
  const ajp::Refusal percent =
      refusal_of(chain("chain_1_model_1_name=2x_m\nchain_1_model_1_resize_factor_before_upscale=50\n"));
  CHECK(percent.code == RefusalCode::ImportUnsupported);
  CHECK(percent.detail.starts_with("ligne 6 : chain_1_model_1_resize_factor_before_upscale"));

  CHECK(refusal_of(chain("chain_1_model_1_name=2x_a\nchain_1_model_2_name=2x_b\n"
                         "chain_1_model_2_resize_height_before_upscale=720\n"))
            .code == RefusalCode::ImportUnsupported);
  CHECK(refusal_of(chain("chain_1_model_1_name=2x_m\nchain_1_model_1_resize_height_before_upscale=100\n"))
            .code == RefusalCode::FieldOutOfRange);
  CHECK(refusal_of(chain("chain_1_model_1_name=16x_m\n")).code == RefusalCode::FieldOutOfRange);
  CHECK(
      refusal_of(chain("chain_1_rife=yes\nchain_1_rife_model=abc\nchain_1_rife_factor_numerator=2\n")).code ==
      RefusalCode::FieldType);
  CHECK(refusal_of(chain("chain_1_rife=yes\nchain_1_rife_factor_denominator=2\n")).code ==
        RefusalCode::FieldOutOfRange);
  CHECK(refusal_of(chain("chain_1_rife=yes\nchain_1_rife_factor_numerator=9\n")).code ==
        RefusalCode::FieldOutOfRange);
  CHECK(refusal_of(chain("chain_1_rife=maybe\n")).code == RefusalCode::FieldType);
  CHECK(
      refusal_of(chain("chain_1_model_1_name=1x_m\nchain_1_model_2_name=1x_m\nchain_1_model_3_name=1x_m\n"
                       "chain_1_model_4_name=1x_m\nchain_1_model_5_name=1x_m\nchain_1_model_6_name=1x_m\n"
                       "chain_1_model_7_name=1x_m\nchain_1_model_8_name=1x_m\nchain_1_model_9_name=1x_m\n"
                       "chain_1_model_10_name=1x_m\nchain_1_model_11_name=1x_m\nchain_1_model_12_name=1x_m\n"
                       "chain_1_model_13_name=1x_m\nchain_1_model_14_name=1x_m\nchain_1_model_15_name=1x_m\n"
                       "chain_1_model_16_name=1x_m\nchain_1_model_17_name=1x_m\n"))
          .code == RefusalCode::TooManySteps);
  CHECK(refusal_of(chain("chain_1_model_1_name=4x_m\nchain_1_rife=yes\nchain_1_rife_factor_numerator=8\n"))
            .code == RefusalCode::FactorProductTooLarge);
}

TEST_CASE("Import INI : bornes de résolution et de fps", "[ini_import]") {
  const auto rule_of = [](const std::string& keys) {
    const ImportResult result = imported(chain(keys + "chain_1_model_1_name=2x_m\n"));
    return result.profile.profiles[0]->rules[0];
  };
  CHECK(rule_of("chain_1_max_resolution=1920x0\n").max_pixels == limits::kPixels.hi);
  CHECK(rule_of("chain_1_max_resolution=infxinf\n").max_pixels == limits::kPixels.hi);
  CHECK(rule_of("chain_1_max_resolution=\n").max_pixels == limits::kPixels.hi);
  CHECK(rule_of("chain_1_min_fps=23.976\nchain_1_max_fps=inf\n").min_fps == 23.976);
  CHECK(rule_of("chain_1_max_fps=inf\n").max_fps == limits::kFps.hi);

  CHECK(refusal_of(chain("chain_1_min_resolution=infx720\n")).code == RefusalCode::FieldOutOfRange);
  CHECK(refusal_of(chain("chain_1_min_resolution=1920*1080\n")).code == RefusalCode::FieldType);
  CHECK(refusal_of(chain("chain_1_max_resolution=99999x99999\n")).code == RefusalCode::FieldOutOfRange);
  CHECK(refusal_of(chain("chain_1_max_fps=2000\n")).code == RefusalCode::FieldOutOfRange);
  CHECK(refusal_of(chain("chain_1_min_fps=vite\n")).code == RefusalCode::FieldType);
}

TEST_CASE("Import INI : étapes ordonnées refusées ou retirées", "[ini_import]") {
  const std::string head = "chain_1_pipeline_version=1\n";
  const auto stage = [&](int count, const std::string& keys) {
    return chain(head + "chain_1_stage_count=" + std::to_string(count) + "\n" + keys);
  };
  CHECK(refusal_of(stage(1, "chain_1_stage_1_kind=interpolate\nchain_1_stage_1_method=nvidia-fruc\n")).code ==
        RefusalCode::ImportUnsupported);
  CHECK(refusal_of(
            stage(2,
                  "chain_1_stage_1_kind=upscale\nchain_1_stage_1_method=onnx\nchain_1_stage_1_model=2x_m\n"
                  "chain_1_stage_2_kind=resize\nchain_1_stage_2_method=area\n"
                  "chain_1_stage_2_target_height=720\n"))
            .code == RefusalCode::ImportUnsupported);
  CHECK(refusal_of(stage(1,
                         "chain_1_stage_1_kind=resize\nchain_1_stage_1_method=area\n"
                         "chain_1_stage_1_target_percent=50\n"))
            .code == RefusalCode::ImportUnsupported);
  CHECK(refusal_of(stage(1, "chain_1_stage_1_kind=upscale\nchain_1_stage_1_method=onnx\n")).code ==
        RefusalCode::FieldMissing);
  CHECK(refusal_of(stage(1,
                         "chain_1_stage_1_kind=interpolate\nchain_1_stage_1_method=rife\n"
                         "chain_1_stage_1_model=426\nchain_1_stage_1_factor_numerator=2\n"
                         "chain_1_stage_1_policy=refine\n"))
            .code == RefusalCode::FieldOutOfRange);
  CHECK(refusal_of(stage(17, "")).code == RefusalCode::TooManySteps);
  CHECK(refusal_of(chain("chain_1_pipeline_version=2\nchain_1_stage_count=0\n")).code ==
        RefusalCode::FormatVersion);
  CHECK(refusal_of(chain("chain_1_pipeline_version=1\n")).code == RefusalCode::FieldMissing);
  CHECK(refusal_of(chain("chain_1_stage_count=1\nchain_1_stage_1_kind=upscale\n")).code ==
        RefusalCode::FieldMissing);
  CHECK(refusal_of(stage(1, "chain_1_stage_1_kind=hdr\nchain_1_stage_1_method=mpv\n") +
                   "chain_2_pipeline_version=1\nchain_2_stage_count=1\nchain_2_stage_1_kind=hdr\n"
                   "chain_2_stage_1_method=mpv\n")
            .code == RefusalCode::FieldOutOfRange);

  const ImportResult disabled = imported(stage(2,
                                               "chain_1_stage_1_kind=upscale\nchain_1_stage_1_method=onnx\n"
                                               "chain_1_stage_1_model=2x_m\nchain_1_stage_1_enabled=no\n"
                                               "chain_1_stage_2_kind=upscale\nchain_1_stage_2_method=onnx\n"
                                               "chain_1_stage_2_model=2x_n\n"));
  REQUIRE(disabled.profile.profiles[0]->rules[0].steps.size() == 1);
  CHECK(disabled.profile.profiles[0]->rules[0].steps[0].model == "2x_n");
  CHECK(has_note(disabled, "désactivée"));

  const ImportResult native = imported(stage(1, "chain_1_stage_1_kind=hdr\nchain_1_stage_1_method=mpv\n"));
  CHECK(native.profile.profiles[0]->hdr == HdrMode::Native);
  CHECK(native.profile.profiles[0]->rules[0].steps.empty());
}

TEST_CASE("Import INI : un agrandissement ordonné garde sa réduction attachée", "[ini_import]") {
  // Normalize (PlatformServices.cs:421-443) sépare resize_*_before_upscale de son modèle.
  const std::string head = "chain_1_pipeline_version=1\n";
  const auto upscale = [](int j, const std::string& model) {
    const std::string p = std::format("chain_1_stage_{}_", j);
    return p + "kind=upscale\n" + p + "method=onnx\n" + p + "model=" + model + "\n";
  };
  const auto steps = steps_of(chain(head + "chain_1_stage_count=1\n" + upscale(1, "2x_m") +
                                    "chain_1_stage_1_resize_height_before_upscale=720\n"));
  REQUIRE(steps.size() == 2);
  CHECK(steps[0] == Step{.kind = StepKind::Downscale, .filter = DownscaleFilter::Spline36, .height = 720});
  CHECK(steps[1] == Step{.kind = StepKind::Upscale, .model = "2x_m", .factor = {2, 1}});

  const ImportResult bilinear = imported(chain(head + "chain_1_stage_count=1\n" + upscale(1, "2x_m") +
                                               "chain_1_stage_1_runtime=vulkan\n"
                                               "chain_1_stage_1_resize_height_before_upscale=720\n"));
  CHECK(has_note(bilinear, "bilinéaire du moteur vulkan"));
  CHECK_FALSE(has_note(bilinear, "sans équivalent"));

  // Une réduction qui n'est pas en tête n'a pas d'équivalent en v5.
  CHECK(refusal_of(chain(head + "chain_1_stage_count=2\n" + upscale(1, "2x_a") + upscale(2, "2x_b") +
                         "chain_1_stage_2_resize_height_before_upscale=720\n"))
            .code == RefusalCode::ImportUnsupported);
  CHECK(refusal_of(chain(head + "chain_1_stage_count=1\n" + upscale(1, "2x_m") +
                         "chain_1_stage_1_resize_factor_before_upscale=50\n"))
            .code == RefusalCode::ImportUnsupported);
}

TEST_CASE("Import INI : l'étape HDR est la dernière active, et la seule du profil", "[ini_import]") {
  const std::string hdr_then_upscale =
      "chain_1_pipeline_version=1\nchain_1_stage_count=2\nchain_1_stage_1_kind=hdr\nchain_1_stage_1_method="
      "mpv\n"
      "chain_1_stage_2_kind=upscale\nchain_1_stage_2_method=onnx\nchain_1_stage_2_model=2x_m\n";
  const ajp::Refusal after = refusal_of(chain(hdr_then_upscale));
  CHECK(after.code == RefusalCode::StepOrder);
  CHECK(after.detail.starts_with("ligne 9 : chain_1_stage_2_kind"));

  // Une étape désactivée après HDR ne compte pas, comme dans Validate.
  const ImportResult disabled = imported(chain(hdr_then_upscale + "chain_1_stage_2_enabled=no\n"));
  CHECK(disabled.profile.profiles[0]->hdr == HdrMode::Native);
  CHECK(disabled.profile.profiles[0]->rules[0].steps.empty());

  const ajp::Refusal second =
      refusal_of(chain("chain_1_pipeline_version=1\nchain_1_stage_count=1\nchain_1_stage_1_kind=hdr\nchain_1_"
                       "stage_1_method=mpv\n"
                       "chain_2_pipeline_version=1\nchain_2_stage_count=1\nchain_2_stage_1_kind=hdr\n"
                       "chain_2_stage_1_method=nvidia-truehdr\n"));
  CHECK(second.code == RefusalCode::FieldOutOfRange);
  CHECK(second.detail.starts_with("ligne 11 : chain_2_stage_1_kind"));
}

TEST_CASE("Import INI : fenêtre temporelle, facteur du nom et hauteur extrême", "[ini_import]") {
  // frames : entier de 0 à 64 (ValidateStageOptions) ; une fenêtre écrite est annoncée.
  CHECK(refusal_of(chain("chain_1_model_1_name=2x_m\nchain_1_model_1_frames=abc\n")).code ==
        RefusalCode::FieldType);
  CHECK(refusal_of(chain("chain_1_model_1_name=2x_m\nchain_1_model_1_frames=65\n")).code ==
        RefusalCode::FieldOutOfRange);
  const ImportResult window = imported(chain("chain_1_model_1_name=2x_m\nchain_1_model_1_frames=5\n"));
  CHECK(has_note(window, "chain_1_model_1_frames = 5"));
  CHECK_FALSE(has_note(imported(chain("chain_1_model_1_name=2x_m\nchain_1_model_1_frames=1\n")), "frames"));

  // ModelScale : « 0x_ » n'est pas un facteur, il vaut 2 comme sans préfixe.
  CHECK(steps_of(chain("chain_1_model_1_name=0x_m\n"))[0].factor == Ratio{2, 1});
  CHECK(steps_of(chain("chain_1_model_1_name=3X-m\n"))[0].factor == Ratio{3, 1});

  // Une hauteur hors de l'int64 est refusée avant toute conversion.
  CHECK(refusal_of(chain("chain_1_model_1_name=2x_m\nchain_1_model_1_resize_height_before_upscale=1e300\n"))
            .code == RefusalCode::FieldOutOfRange);
  CHECK(refusal_of(chain("chain_1_model_1_name=2x_m\nchain_1_model_1_resize_height_before_upscale=720.5\n"))
            .code == RefusalCode::FieldOutOfRange);
}

TEST_CASE("Import INI : versions, slots et profil de démarrage", "[ini_import]") {
  const ajp::Refusal newer = refusal_of("[global]\nconfig_version=5\n");
  CHECK(newer.code == RefusalCode::FormatVersion);
  CHECK(newer.detail.starts_with("ligne 2 : config_version"));
  CHECK(refusal_of("[global]\nconfig_version=deux\n").code == RefusalCode::FieldType);
  CHECK(refusal_of("[global]\nconfig_version=0\n").code == RefusalCode::FieldOutOfRange);
  CHECK(refusal_of("[slot_12]\nprofile_name=x\n").code == RefusalCode::FieldOutOfRange);
  CHECK(refusal_of("[slot_x]\nprofile_name=x\n").code == RefusalCode::FieldOutOfRange);
  CHECK(refusal_of("[global]\nrife_engine=turbo\n").code == RefusalCode::FieldOutOfRange);

  const ImportResult off = imported(
      "[global]\ndefault_slot=0\n[slot_4]\nprofile_name=A\n[slot_6]\nprofile_name=B\n"
      "[slot_1001]\nprofile_name=Quality\n[extra]\nk=v\n");
  CHECK(off.profile.start_profile == 4);
  CHECK(off.profile.profiles[3]->name == "A");
  CHECK_FALSE(off.profile.profiles[0]);
  CHECK(has_note(off, "default_slot = 0"));
  CHECK(has_note(off, "slot_1001"));
  CHECK(has_note(off, "[extra]"));

  // Sans profile_name, le nom que le fork et le gestionnaire affichent.
  CHECK(imported("[slot_3]\nchain_1_model_1_name=2x_m\n").profile.profiles[2]->name == "New Profile");

  const ImportResult empty = imported("");
  CHECK(empty.profile == ProfileFile{});
  CHECK(empty.notes.empty());
  CHECK(parse_profile(empty.toml) == ProfileFile{});
}

TEST_CASE("Import INI : lignes, commentaires, BOM et clés inconnues", "[ini_import]") {
  const ImportResult crlf = imported(
      "\xEF\xBB\xBF; commentaire\r\n[GLOBAL]\r\nDefault_Slot = 1\r\n[slot_1]\r\n"
      "# autre\r\nprofile_name = Mon profil \r\nchain_1_model_1_name=2x_m\r\n"
      "chain_1_model_1_mystery=1\r\nhdr_stage_controlled=yes\r\nchain__rife=no\r\n");
  CHECK(crlf.profile.start_profile == 1);
  CHECK(crlf.profile.profiles[0]->name == "Mon profil");
  CHECK(has_note(crlf, "chain_1_model_1_mystery"));
  CHECK(has_note(crlf, "chain__rife"));
  CHECK_FALSE(has_note(crlf, "hdr_stage_controlled"));

  const ImportResult repeated = imported(chain("profile_name=Second\n"));
  CHECK(repeated.profile.profiles[0]->name == "Second");
  CHECK(has_note(repeated, "répétée"));

  const ajp::Refusal no_equals = refusal_of("[global]\njuste du texte\n");
  CHECK(no_equals.code == RefusalCode::FieldType);
  CHECK(no_equals.detail.starts_with("ligne 2"));
  CHECK(refusal_of("cle=valeur\n").code == RefusalCode::FieldType);
  CHECK(refusal_of("[global]\n=valeur\n").code == RefusalCode::FieldType);
}

TEST_CASE("Import INI : depuis un fichier", "[ini_import]") {
  const fs::path dir = fs::current_path() / "test-tmp" / "ini-file";
  fs::remove_all(dir);
  fs::create_directories(dir);
  {
    std::ofstream out(dir / "animejanai.conf", std::ios::binary);
    out << kLegacy;
  }
  auto result = import_ini_file(dir / "animejanai.conf");
  REQUIRE(result);
  CHECK(result->profile == imported(kLegacy).profile);
  auto missing = import_ini_file(dir / "absent.conf");
  REQUIRE_FALSE(missing);
  CHECK(missing.error().code == RefusalCode::FileUnreadable);
}
