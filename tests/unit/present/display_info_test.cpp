// Détection de l'écran sans écran : rafraîchissement nominal, priorités et provenance de
// chaque champ, blanc SDR proposé, bornes refusées ; espace de sortie et repli scRGB ; écart
// entre l'horloge audio et celle de l'écran.
#include "present/display_info.h"

#include <optional>
#include <string_view>

#include <catch2/catch_test_macros.hpp>

#include "core/debug_flags.h"
#include "present/display_clock.h"
#include "present/output.h"
#include "present/refresh.h"

using ajp::ClockSnapshot;
using ajp::RefusalCode;
using ajp::core::DisplayInfo;
using ajp::core::OutputSpace;
using ajp::core::Provenance;
using ajp::core::Ratio;
using ajp::present::detect_display;
using ajp::present::DisplayOverrides;
using ajp::present::nominal_refresh;
using ajp::present::OutputHint;
using ajp::present::SurfaceTransfer;
using ajp::present::SystemDisplay;

TEST_CASE("rafraîchissement : la mesure ramenée à sa valeur nominale") {
  CHECK(nominal_refresh(Ratio{164998, 1000}) == Ratio{165, 1});
  CHECK(nominal_refresh(164.9987) == Ratio{165, 1});
  CHECK(nominal_refresh(Ratio{60000, 1001}) == Ratio{60000, 1001});
  CHECK(nominal_refresh(59.9401) == Ratio{60000, 1001});
  CHECK(nominal_refresh(Ratio{59940, 1000}) == Ratio{60000, 1001});
  CHECK(nominal_refresh(60.0) == Ratio{60, 1});
  CHECK(nominal_refresh(143.856) == Ratio{144000, 1001});
  // Ni entier ni NTSC : la mesure au millihertz.
  CHECK(nominal_refresh(74.973) == Ratio{74973, 1000});
  CHECK(nominal_refresh(0.5).error().code == RefusalCode::FieldOutOfRange);
  CHECK(nominal_refresh(1001.0).error().code == RefusalCode::FieldOutOfRange);
  CHECK(nominal_refresh(Ratio{0, 1}).error().code == RefusalCode::FieldOutOfRange);
}

namespace {

// L'écran de ce PC tel que SDL3 le lit sous KDE Plasma en HDR.
SystemDisplay qd_oled() {
  return {.name = "MSI MAG 272UP",
          .width = 3840,
          .height = 2160,
          .refresh = Ratio{164998, 1000},
          .hdr = true,
          .sdr_white_nits = 240.0,
          .peak_nits = 1000.0,
          .unknown = {"VRR"}};
}

}  // namespace

TEST_CASE("écran lu par le système : provenance système, blanc SDR seulement proposé") {
  const auto detected = detect_display(qd_oled(), {}, std::nullopt).value();
  const DisplayInfo& d = detected.info;
  CHECK(d.width == 3840);
  CHECK(d.height == 2160);
  CHECK(d.refresh_hz == 165.0);
  CHECK(detected.refresh == Ratio{165, 1});
  CHECK(d.hdr);
  CHECK(d.space == OutputSpace::Pq);
  CHECK(d.peak_nits == 1000.0);
  CHECK(d.provenance.size == Provenance::System);
  CHECK(d.provenance.refresh == Provenance::System);
  CHECK(d.provenance.peak == Provenance::System);
  // Jamais appliqué en silence : le blanc reste celui de BT.2408, le système le propose.
  CHECK(d.sdr_white_nits == 203.0);
  CHECK(d.provenance.sdr_white == Provenance::Default);
  CHECK(detected.proposed_sdr_white_nits == 240.0);
  CHECK_FALSE(d.vrr);
  CHECK(d.provenance.vrr == Provenance::Default);
  CHECK(detected.unknown.size() == 1);
}

TEST_CASE("écran SDR : la crête du système n'est pas lue") {
  SystemDisplay sdr = qd_oled();
  sdr.hdr = false;
  sdr.peak_nits = 80.0;
  const auto detected = detect_display(sdr, {}, std::nullopt).value();
  CHECK(detected.info.peak_nits == 203.0);
  CHECK(detected.info.provenance.peak == Provenance::Default);
  CHECK(detected.info.space == OutputSpace::Srgb);
}

TEST_CASE("chaque champ se force dans le profil, --display l'emporte") {
  const DisplayOverrides profile{.width = 2560,
                                 .height = 1440,
                                 .refresh_hz = 120.0,
                                 .hdr = false,
                                 .vrr = true,
                                 .peak_nits = 800.0,
                                 .min_nits = 0.0005,
                                 .sdr_white_nits = 240.0,
                                 .space = OutputSpace::ScRgb};
  const auto forced = detect_display(qd_oled(), profile, std::nullopt).value();
  const DisplayInfo& d = forced.info;
  CHECK(d.width == 2560);
  CHECK(d.refresh_hz == 120.0);
  CHECK_FALSE(d.hdr);
  CHECK(d.vrr);
  CHECK(d.peak_nits == 800.0);
  CHECK(d.min_nits == 0.0005);
  CHECK(d.sdr_white_nits == 240.0);
  CHECK(d.space == OutputSpace::ScRgb);
  const ajp::core::DisplayProvenance all_profile{
      Provenance::Profile, Provenance::Profile, Provenance::Profile, Provenance::Profile,
      Provenance::Profile, Provenance::Profile, Provenance::Profile, Provenance::Profile};
  CHECK(d.provenance == all_profile);

  const std::string_view args[] = {"--display=1920x1080@60:hdr"};
  const auto line = ajp::core::parse_command_line(args);
  REQUIRE(line);
  const auto overridden = detect_display(qd_oled(), profile, line->debug.display).value();
  CHECK(overridden.info.width == 1920);
  CHECK(overridden.info.refresh_hz == 60.0);
  CHECK(overridden.info.hdr);
  CHECK(overridden.info.space == OutputSpace::Pq);
  CHECK(overridden.info.provenance.size == Provenance::CommandLine);
  CHECK(overridden.info.provenance.space == Provenance::CommandLine);
  // Ce que --display ne dit pas garde le profil.
  CHECK(overridden.info.vrr);
  CHECK(overridden.info.provenance.vrr == Provenance::Profile);
}

TEST_CASE("valeur hors bornes refusée, avec sa provenance") {
  const auto peak = detect_display(qd_oled(), {.peak_nits = 20'000.0}, std::nullopt);
  REQUIRE_FALSE(peak);
  CHECK(peak.error().code == RefusalCode::FieldOutOfRange);
  CHECK(peak.error().detail.find("profil") != std::string::npos);

  CHECK(detect_display(qd_oled(), {.refresh_hz = 2000.0}, std::nullopt).error().code ==
        RefusalCode::FieldOutOfRange);
  CHECK(detect_display(qd_oled(), {.width = 0, .height = 1080}, std::nullopt).error().code ==
        RefusalCode::FieldOutOfRange);
  CHECK(detect_display(qd_oled(), {.width = 1920}, std::nullopt).error().code ==
        RefusalCode::FieldOutOfRange);
  // Noir au-dessus de la crête : refusé, pas corrigé.
  SystemDisplay inverted = qd_oled();
  inverted.min_nits = 2000.0;
  CHECK(detect_display(inverted, {}, std::nullopt).error().code == RefusalCode::FieldOutOfRange);
  // Un rafraîchissement forcé à 0 rend l'écran inconnu pour la cadence.
  const auto unknown = detect_display(qd_oled(), {.refresh_hz = 0.0}, std::nullopt).value();
  CHECK(unknown.info.refresh_hz == 0.0);
  CHECK_FALSE(unknown.refresh);
}

TEST_CASE("sortie : HDR10 demandé, repli scRGB qui coupe le grade, puis SDR") {
  using ajp::present::classify_output;
  using ajp::present::fallback_hint;
  using ajp::present::first_hint;
  CHECK(first_hint({.hdr = true}) == OutputHint::Pq);
  CHECK(first_hint({.hdr = false}) == OutputHint::Sdr);
  CHECK(first_hint({.hdr = true, .space = OutputSpace::ScRgb}) == OutputHint::ScRgb);

  const auto pq = classify_output(OutputHint::Pq, SurfaceTransfer::Pq);
  CHECK(pq.space == OutputSpace::Pq);
  CHECK(pq.grade);
  CHECK_FALSE(pq.hdr_lost);

  CHECK(fallback_hint(OutputHint::Pq, SurfaceTransfer::Other) == OutputHint::ScRgb);
  const auto scrgb = classify_output(OutputHint::ScRgb, SurfaceTransfer::Linear);
  CHECK(scrgb.space == OutputSpace::ScRgb);
  CHECK_FALSE(scrgb.grade);
  CHECK(scrgb.hdr_lost);

  CHECK_FALSE(fallback_hint(OutputHint::ScRgb, SurfaceTransfer::Other));
  const auto sdr = classify_output(OutputHint::ScRgb, SurfaceTransfer::Other);
  CHECK(sdr.space == OutputSpace::Srgb);
  CHECK(sdr.hdr_lost);
  const auto plain = classify_output(OutputHint::Sdr, SurfaceTransfer::Other);
  CHECK_FALSE(plain.hdr_requested);
  CHECK_FALSE(plain.hdr_lost);
}

TEST_CASE("écart de l'horloge audio sur l'horloge de l'écran") {
  using ajp::present::audio_offset_us;
  const ClockSnapshot display{.media_us = 1'000'000, .host_us = 0, .rate = 1.001, .epoch = 2, .valid = true};
  const ClockSnapshot audio{.media_us = 1'000'500, .host_us = 0, .rate = 1.0, .epoch = 2, .valid = true};
  CHECK(audio_offset_us(audio, display, 0) == 500);
  CHECK(audio_offset_us(audio, display, 1'000'000) == 500 - 1000);
  ClockSnapshot other_epoch = audio;
  other_epoch.epoch = 1;
  CHECK_FALSE(audio_offset_us(other_epoch, display, 0));
  CHECK_FALSE(audio_offset_us(audio, ClockSnapshot{}, 0));

  ajp::present::DisplayClock clock;
  CHECK_FALSE(clock.snapshot().valid);
  clock.publish(display);
  CHECK(clock.snapshot().media_us == 1'000'000);
}
