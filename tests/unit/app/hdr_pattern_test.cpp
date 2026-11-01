#include "app/play/hdr_pattern.h"

#include <cstdlib>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "app/play/display_edid.h"
#include "app/play/settings.h"

using ajp::app::play::HdrPattern;
using ajp::app::play::OutputSize;
using ajp::app::play::pattern_layout;
using ajp::app::play::pq_of_nits;
using Catch::Approx;

TEST_CASE("mire HDR : le PQ des nits suit ST 2084", "[hdr_pattern]") {
  CHECK(pq_of_nits(0.0) == Approx(0.0).margin(1e-6));
  CHECK(pq_of_nits(100.0) == Approx(0.5081).margin(1e-3));
  CHECK(pq_of_nits(1000.0) == Approx(0.7518).margin(1e-3));
  CHECK(pq_of_nits(10000.0) == Approx(1.0).margin(1e-9));
  CHECK(pq_of_nits(20000.0) == Approx(1.0).margin(1e-9));  // borné au maximum du PQ
}

TEST_CASE("mire HDR : un cadre de 10 % de l'écran, le carré au milieu", "[hdr_pattern]") {
  const auto layout = pattern_layout(OutputSize{3840, 2160}, HdrPattern{});
  const auto& f = layout.frame;
  const double area = static_cast<double>(f.x1 - f.x0) * (f.y1 - f.y0) / (3840.0 * 2160.0);
  CHECK(area == Approx(0.10).margin(0.002));
  CHECK(std::abs(f.y0 + f.y1 - 2160) <= 1);  // centré en hauteur
  CHECK(f.x0 >= 0);
  const auto& s = layout.square;
  CHECK(s.x1 - s.x0 == s.y1 - s.y0);  // un carré
  CHECK(s.x0 > f.x0);
  CHECK(s.x1 < f.x1);
  CHECK(s.y0 > f.y0);
  CHECK(s.y1 < f.y1);
  CHECK(std::abs((s.x0 + s.x1) - (f.x0 + f.x1)) <= 2);
}

TEST_CASE("mire HDR : la fenêtre à 50 % reste dans l'écran", "[hdr_pattern]") {
  HdrPattern p;
  p.area = 0.50;
  const auto f = pattern_layout(OutputSize{1920, 1080}, p).frame;
  CHECK(f.x0 >= 0);
  CHECK(f.x1 <= 1920);
  CHECK(f.y0 >= 0);
  CHECK(f.y1 <= 1080);
}

TEST_CASE("calibration : le pic mesuré va à son écran et survit à play.toml", "[hdr_pattern]") {
  using namespace ajp::app::play;
  PlaySettings s;
  CHECK(auto_peak_for(s, "CND-8CD7", 617) == 617);  // sans calibration : l'EDID
  set_calibrated_peak(s, "CND-8CD7", 800);
  set_calibrated_peak(s, "GSM-5BBF", 1000);
  set_calibrated_peak(s, "", 900);  // écran inconnu : rien n'est gardé
  CHECK(auto_peak_for(s, "CND-8CD7", 617) == 800);
  CHECK(auto_peak_for(s, "AUS-27F1", 617) == 617);  // un autre écran garde son EDID
  set_calibrated_peak(s, "CND-8CD7", 820);          // une nouvelle mesure remplace l'ancienne
  CHECK(s.display_peaks.size() == 2);
  const auto back = parse_settings(format_settings(s));
  REQUIRE(back.has_value());
  CHECK(calibrated_peak(*back, "CND-8CD7") == 820);
  CHECK(calibrated_peak(*back, "GSM-5BBF") == 1000);
  set_calibrated_peak(s, "CND-8CD7", 0);  // oubliée
  CHECK(calibrated_peak(s, "CND-8CD7") == 0);
}

TEST_CASE("calibration : la clé d'un écran", "[hdr_pattern]") {
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
