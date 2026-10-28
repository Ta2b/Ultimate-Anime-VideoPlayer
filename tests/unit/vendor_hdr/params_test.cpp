#include <limits>

#include <catch2/catch_test_macros.hpp>

#include "vendor_hdr/true_hdr.h"

using ajp::ColorMatrix;
using ajp::Primaries;
using ajp::RefusalCode;
using ajp::Transfer;
using namespace ajp::vendor_hdr;

TEST_CASE("réglages : valeurs par défaut du plan", "[params]") {
  const TrueHdrParams params;
  CHECK(params.contrast == 100);
  CHECK(params.saturation == 100);
  CHECK(params.middle_gray == 50);
  CHECK(params.peak_nits == 1000);
  CHECK(validate(params));
}

TEST_CASE("réglages : plages incluses", "[params]") {
  CHECK(limits::kContrast.lo == 0);
  CHECK(limits::kContrast.hi == 200);
  CHECK(limits::kSaturation.lo == 0);
  CHECK(limits::kSaturation.hi == 200);
  CHECK(limits::kMiddleGray.lo == 10);
  CHECK(limits::kMiddleGray.hi == 100);
  CHECK(limits::kPeakNits.lo == 400);
  CHECK(limits::kPeakNits.hi == 2000);

  CHECK(validate({.contrast = 0, .saturation = 0, .middle_gray = 10, .peak_nits = 400}));
  CHECK(validate({.contrast = 200, .saturation = 200, .middle_gray = 100, .peak_nits = 2000}));
}

TEST_CASE("réglages : une valeur hors plage est refusée, jamais corrigée", "[params]") {
  struct Case {
    TrueHdrParams params;
    const char* detail;
  };
  const Case cases[] = {
      {{.contrast = -1}, "truehdr.contrast = -1, hors de 0-200"},
      {{.contrast = 201}, "truehdr.contrast = 201, hors de 0-200"},
      {{.saturation = 201}, "truehdr.saturation = 201, hors de 0-200"},
      {{.middle_gray = 9}, "truehdr.middle_gray = 9, hors de 10-100"},
      {{.middle_gray = 101}, "truehdr.middle_gray = 101, hors de 10-100"},
      {{.peak_nits = 399}, "truehdr.peak = 399, hors de 400-2000"},
      {{.peak_nits = 2001}, "truehdr.peak = 2001, hors de 400-2000"},
  };
  for (const Case& item : cases) {
    auto valid = validate(item.params);
    REQUIRE_FALSE(valid);
    CHECK(valid.error().code == RefusalCode::FieldOutOfRange);
    CHECK(valid.error().detail == item.detail);
  }
}

TEST_CASE("pic : luminance de l'écran bornée, 1000 si inconnue", "[params]") {
  CHECK(peak_from_display(std::nullopt) == 1000);
  CHECK(peak_from_display(0.0) == 1000);
  CHECK(peak_from_display(-50.0) == 1000);
  CHECK(peak_from_display(std::numeric_limits<double>::quiet_NaN()) == 1000);
  CHECK(peak_from_display(std::numeric_limits<double>::infinity()) == 1000);
  CHECK(peak_from_display(250.0) == 400);
  CHECK(peak_from_display(400.0) == 400);
  CHECK(peak_from_display(799.6) == 800);
  CHECK(peak_from_display(2000.0) == 2000);
  CHECK(peak_from_display(10000.0) == 2000);
}

TEST_CASE("pic : crête vue du contenu sous KWin, défauts indépendants du SDK", "[params]") {
  // Écran de référence (display.md) : marge HDR 1000 / 265 relative au blanc de KWin, rapportée
  // au blanc 203 nits ; écran assombri par KDE, marge 1000 / 83, bornée au plus haut.
  CHECK(peak_from_display(1000.0 / 265.0 * 203.0) == 766);
  CHECK(peak_from_display(1000.0 / 83.0 * 203.0) == 2000);
  // Défauts du plan : pic 1000 (650 dans le SDK Maxine), débandage actif (le SDK aussi).
  CHECK(TrueHdrParams{}.peak_nits == 1000);
  CHECK(TrueHdrParams{}.debanding);
  CHECK(TrueHdrParams{.debanding = false} != TrueHdrParams{});
}

TEST_CASE("configuration : taille, source SDR et réglages", "[params]") {
  TrueHdrConfig config{.width = 1920, .height = 1080};
  CHECK(check(config));

  SECTION("côté trop petit ou trop grand") {
    config.width = 1;
    CHECK(check(config).error().code == RefusalCode::FieldOutOfRange);
    config.width = 16385;
    CHECK(check(config).error().code == RefusalCode::FieldOutOfRange);
  }
  SECTION("source déjà HDR") {
    config.source.transfer = Transfer::Pq;
    CHECK(check(config).error().code == RefusalCode::HdrUnavailable);
  }
  SECTION("source BT.2020") {
    config.source.primaries = Primaries::Bt2020;
    CHECK(check(config).error().code == RefusalCode::HdrUnavailable);
    config.source.primaries = Primaries::Bt709;
    config.source.matrix = ColorMatrix::Bt2020Ncl;
    CHECK(check(config).error().code == RefusalCode::HdrUnavailable);
  }
  SECTION("source SD en BT.601") {
    config.source.matrix = ColorMatrix::Bt601;
    config.source.primaries = Primaries::Bt601_525;
    config.source.transfer = Transfer::Srgb;
    CHECK(check(config));
  }
  SECTION("réglage hors plage") {
    config.params.middle_gray = 5;
    CHECK(check(config).error().detail == "truehdr.middle_gray = 5, hors de 10-100");
  }
}
