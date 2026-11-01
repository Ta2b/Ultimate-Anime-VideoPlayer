#include <algorithm>
#include <cstdint>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "app/play/calibration_images.h"
#include "app/play/display_edid.h"
#include "app/play/i18n.h"
#include "app/play/settings.h"
#include "app/play/system_peak.h"

using Catch::Approx;

namespace {

// Extrait de ~/.config/kwinoutputconfig.json (Plasma 6) : deux écrans déjà branchés sur DP-1, un seul en HDR.
constexpr std::string_view kKwin = R"([
  {"name": "outputs", "data": [
    {"connectorName": "DP-1", "edidIdentifier": "AUS 10161 0 0 2020 0", "highDynamicRange": false,
     "sdrBrightness": 200},
    {"connectorName": "HDMI-A-1", "highDynamicRange": true, "maxPeakBrightnessOverride": null,
     "referenceLuminance": 250},
    {"connectorName": "DP-1", "edidIdentifier": "MSI 15052 0 0 2024 0", "highDynamicRange": true,
     "maxPeakBrightnessOverride": 1015, "sdrBrightness": 265, "iccProfilePath": "a{b}.icc"}
  ]},
  {"name": "setups", "data": [{"outputs": [{"outputIndex": 0}]}]}
])";

}  // namespace

TEST_CASE("pic du système : la sortie de KWin en HDR, sa crête et son blanc", "[calibration]") {
  using namespace ajp::app::play;
  const auto dp = parse_kwin_output(kKwin, "DP-1");
  REQUIRE(dp.has_value());
  CHECK(dp->peak_override == Approx(1015.0));
  CHECK(dp->reference == Approx(265.0));
  const auto hdmi = parse_kwin_output(kKwin, "HDMI-A-1");
  REQUIRE(hdmi.has_value());
  CHECK_FALSE(hdmi->peak_override.has_value());  // null : pas de crête à la main
  CHECK(hdmi->reference == Approx(250.0));
  CHECK_FALSE(parse_kwin_output(kKwin, "DP-2").has_value());
  CHECK_FALSE(parse_kwin_output(kKwin, "").has_value());
  CHECK_FALSE(parse_kwin_output("not json", "DP-1").has_value());
}

TEST_CASE("calibration du grade : plans d'anime sombres, sombres avec éclat et clairs, guidés et traduits",
          "[calibration]") {
  using namespace ajp::app::play;
  std::set<int> ids;
  std::set<std::int64_t> instants;
  int previous = -1;
  for (const CalibrationStep step : {CalibrationStep::Dark, CalibrationStep::Glow, CalibrationStep::Bright}) {
    CHECK(calibration_count(step) >= 2);
    for (int i = 0; i < calibration_count(step); ++i) {
      const CalibrationImage image = calibration_image(step, i);
      ids.insert(image.id);
      instants.insert(image.at_us);
      CHECK(image.id == previous + 1);  // les étapes se suivent dans la vidéo, sans trou
      previous = image.id;
      // Au milieu de son plan de la vidéo de calibration, loin des coupes.
      CHECK(image.at_us == image.id * kCalibrationSegmentUs + kCalibrationSegmentUs / 2);
      i18n::set_language(i18n::Language::French);
      CHECK(i18n::tr_view(image.title) != image.title);
      CHECK(i18n::tr_view(image.guidance) != image.guidance);
      i18n::set_language(i18n::Language::English);
    }
  }
  // Un index hors de l'étape reste dans l'étape.
  CHECK(calibration_image(CalibrationStep::Glow, 9).id == calibration_image(CalibrationStep::Glow, 1).id);
  CHECK(ids.size() == kCalibrationImages);
  CHECK(instants.size() == kCalibrationImages);
  CHECK(calibration_clip().filename() == "calibration.mkv");
}

TEST_CASE("calibration : la clé d'un écran", "[calibration]") {
  using namespace ajp::app::play;
  DisplayFacts d;
  d.manufacturer = "CND";
  d.product = 0x8CD7;
  d.name = "MAG 272U E16";
  CHECK(display_key(d) == "CND-8CD7");
  d.product = 0;
  CHECK(display_key(d) == "MAG 272U E16");
  CHECK(display_key(DisplayFacts{}).empty());
}

TEST_CASE("calibration du grade : gardée par écran, reprise à l'ouverture si le grade adaptatif est coché",
          "[calibration]") {
  using namespace ajp::app::play;
  PlaySettings s;
  s.true_hdr_gray = 42;
  s.grade_values.shadows = 0.3;
  s.grade_values.vibrance_rtx = 0.08;
  s.grade_values.adaptive_gray = 18.0;
  s.grade_values.adaptive_shadows = 0.6;
  s.grade_values.adaptive_vibrance_cut = 0.55;
  keep_calibrated_grade(s, "CND-8CD7", "Soir");
  keep_calibrated_grade(s, "", "Soir");  // écran inconnu : rien
  REQUIRE(s.display_grades.size() == 1);
  const auto back = parse_settings(format_settings(s));
  REQUIRE(back.has_value());
  REQUIRE(calibrated_grade(*back, "CND-8CD7") != nullptr);
  CHECK(*calibrated_grade(*back, "CND-8CD7") == s.display_grades.front());

  PlaySettings fresh = *back;
  fresh.true_hdr_gray = 50;
  fresh.grade_values.adaptive_vibrance_cut = 0.0;
  fresh.grade_values.adaptive = false;
  CHECK(with_calibrated_grade(fresh, "CND-8CD7").true_hdr_gray == 50);  // adaptatif décoché : rien
  fresh.grade_values.adaptive = true;
  const PlaySettings applied = with_calibrated_grade(fresh, "CND-8CD7");
  CHECK(applied.true_hdr_gray == 42);
  CHECK(applied.grade_values.adaptive_vibrance_cut == Approx(0.55));
  CHECK(with_calibrated_grade(fresh, "GSM-5BBF").true_hdr_gray == 50);  // un autre écran
  forget_calibrated_grade(fresh, "CND-8CD7", "Soir");
  CHECK(calibrated_grade(fresh, "CND-8CD7") == nullptr);
}

TEST_CASE("calibration du grade : plusieurs profils par écran, un seul en service, supprimables",
          "[calibration]") {
  using namespace ajp::app::play;
  PlaySettings s;
  s.grade_values.adaptive = true;
  s.true_hdr_gray = 40;
  keep_calibrated_grade(s, "CND-8CD7", "Jour");
  s.true_hdr_gray = 60;
  keep_calibrated_grade(s, "CND-8CD7", "Nuit");
  keep_calibrated_grade(s, "GSM-5BBF", "Jour");
  CHECK(grade_profiles(s, "CND-8CD7") == std::vector<std::string>{"Jour", "Nuit"});
  REQUIRE(calibrated_grade(s, "CND-8CD7") != nullptr);
  CHECK(calibrated_grade(s, "CND-8CD7")->name == "Nuit");  // le dernier calibré est en service
  REQUIRE(use_calibrated_grade(s, "CND-8CD7", "Jour"));
  CHECK(s.true_hdr_gray == 40);
  CHECK_FALSE(use_calibrated_grade(s, "CND-8CD7", "Inconnu"));
  // Relu à l'identique, profil en service compris.
  const auto back = parse_settings(format_settings(s));
  REQUIRE(back.has_value());
  CHECK(back->display_grades == s.display_grades);
  // Recalibrer garde le nom et la place.
  s.true_hdr_gray = 45;
  keep_calibrated_grade(s, "CND-8CD7", "Jour");
  CHECK(grade_profiles(s, "CND-8CD7").size() == 2);
  CHECK(calibrated_grade(s, "CND-8CD7")->gray == 45);
  // Supprimer celui en service : le suivant le remplace.
  forget_calibrated_grade(s, "CND-8CD7", "Jour");
  REQUIRE(calibrated_grade(s, "CND-8CD7") != nullptr);
  CHECK(calibrated_grade(s, "CND-8CD7")->name == "Nuit");
  CHECK(next_grade_profile_name(s, "CND-8CD7") == "Profile 2");
}

TEST_CASE("calibration du grade : le play.toml de la rc.9 (un grade par écran) devient un profil",
          "[calibration]") {
  using namespace ajp::app::play;
  const auto s = parse_settings(
      "[hdr.display_grades.CND-8CD7]\ngray = 42\nshadows = 0.3\ncolors = 0.08\ngray_bright = 18.0\n"
      "shadows_bright = 0.6\ncolors_cut = 0.55\n");
  REQUIRE(s.has_value());
  REQUIRE(calibrated_grade(*s, "CND-8CD7") != nullptr);
  CHECK(calibrated_grade(*s, "CND-8CD7")->gray == 42);
  CHECK(grade_profiles(*s, "CND-8CD7").size() == 1);
}
