#include <algorithm>
#include <cstdint>
#include <set>
#include <string_view>

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

TEST_CASE("calibration du grade : quatre images d'anime sombres et quatre claires, guidées et traduites",
          "[calibration]") {
  using namespace ajp::app::play;
  std::set<int> ids;
  for (const bool dark : {true, false}) {
    for (int i = 0; i < kCalibrationPerStep; ++i) {
      const CalibrationImage image = calibration_image(dark, i);
      ids.insert(image.id);
      CHECK((image.id < kCalibrationPerStep) == dark);
      CHECK(image.width == 1920);
      CHECK(image.height == 1080);
      REQUIRE(image.rgba.size() ==
              static_cast<std::size_t>(image.width) * static_cast<std::size_t>(image.height) * 4);
      i18n::set_language(i18n::Language::French);
      CHECK(i18n::tr_view(image.title) != image.title);
      CHECK(i18n::tr_view(image.guidance) != image.guidance);
      i18n::set_language(i18n::Language::English);
      // Une vraie image : des pixels sombres et d'autres moins.
      const auto [lo, hi] = std::minmax_element(image.rgba.begin(), image.rgba.end());
      CHECK(*hi - *lo > 10000);
    }
  }
  CHECK(ids.size() == 2 * kCalibrationPerStep);
  // Une scène sombre reste sombre en moyenne, une claire reste claire.
  const auto mean = [](const CalibrationImage& image) {
    double sum = 0.0;
    for (std::size_t i = 0; i < image.rgba.size(); i += 4) sum += image.rgba[i + 1];
    return sum / (static_cast<double>(image.rgba.size()) / 4.0) / 65535.0;
  };
  for (int i = 0; i < kCalibrationPerStep; ++i) {
    CHECK(mean(calibration_image(true, i)) < 0.35);
    CHECK(mean(calibration_image(false, i)) > 0.35);
  }
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
  keep_calibrated_grade(s, "CND-8CD7");
  keep_calibrated_grade(s, "");  // écran inconnu : rien
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
  forget_calibrated_grade(fresh, "CND-8CD7");
  CHECK(calibrated_grade(fresh, "CND-8CD7") == nullptr);
}
