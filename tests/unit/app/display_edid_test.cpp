#include "app/play/display_edid.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <spdlog/spdlog.h>

#include "app/play/display_contract.h"
#include "app/play/display_gpu.h"
#include "app/play/host_facts.h"
#include "app/play/system_peak.h"
#include "app/play/wayland_output.h"
#include "french_ui.h"

using namespace ajp::app::play;
using Catch::Approx;
using Catch::Matchers::WithinAbs;

namespace {

// EDID du QD-OLED 4K de référence (MSI MAG 272U E16, CND 0x8CD7) : bloc de base, CTA-861 (HDR
// statique : PQ, contenu max 1037 nits, moyenne 265, noir code 2), DisplayID 1.x (minutages
// seulement, pas de bloc 0x21).
constexpr std::array<std::uint8_t, 384> kQdOledEdid{
    0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x0d, 0xc4, 0xd7, 0x8c, 0x01, 0x01, 0x01, 0x01, 0x0a,
    0x24, 0x01, 0x04, 0xb5, 0x3b, 0x21, 0x78, 0xfb, 0x00, 0xf5, 0xaf, 0x4e, 0x40, 0xb7, 0x25, 0x0f, 0x50,
    0x54, 0x25, 0x4f, 0x00, 0x3b, 0x80, 0x81, 0x80, 0x95, 0x00, 0xb3, 0x00, 0xd1, 0xc0, 0xd1, 0xfc, 0x01,
    0x01, 0x01, 0x01, 0x4d, 0xd0, 0x00, 0xa0, 0xf0, 0x70, 0x3e, 0x80, 0x30, 0x20, 0x35, 0x00, 0x4e, 0x4d,
    0x21, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00, 0xfd, 0x0c, 0x64, 0xa5, 0x83, 0x83, 0xa0, 0x01, 0x0a, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x00, 0x00, 0x00, 0xfc, 0x00, 0x4d, 0x41, 0x47, 0x20, 0x32, 0x37, 0x32,
    0x55, 0x20, 0x45, 0x31, 0x36, 0x0a, 0x00, 0x00, 0x00, 0xff, 0x00, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x02, 0x81, 0x02, 0x03, 0x36, 0x61, 0x46, 0x03, 0x12, 0x04,
    0x2f, 0x90, 0x3f, 0x23, 0x09, 0x07, 0x07, 0x83, 0x01, 0x00, 0x00, 0xe2, 0x00, 0xea, 0x74, 0x1a, 0x00,
    0x00, 0x03, 0x03, 0x64, 0xa5, 0x00, 0xa0, 0x8c, 0x02, 0x4d, 0x02, 0xa5, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0xe3, 0x05, 0xc0, 0x01, 0xe6, 0x06, 0x05, 0x01, 0x8c, 0x4d, 0x02, 0x56, 0x5e, 0x00, 0xa0, 0xa0,
    0xa0, 0x29, 0x50, 0x30, 0x20, 0x35, 0x00, 0x4e, 0x4d, 0x21, 0x00, 0x00, 0x1a, 0x6f, 0xc2, 0x00, 0xa0,
    0xa0, 0xa0, 0x55, 0x50, 0x30, 0x20, 0x35, 0x00, 0x4e, 0x4d, 0x21, 0x00, 0x00, 0x1a, 0xa4, 0x9c, 0x80,
    0xa0, 0x70, 0x38, 0x59, 0x40, 0x30, 0x20, 0x35, 0x00, 0x4e, 0x4d, 0x21, 0x00, 0x00, 0x1a, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xd3, 0x70, 0x12, 0x79, 0x03, 0x00, 0x03, 0x01, 0x3c, 0x7f, 0x9c, 0x01, 0x04, 0xff, 0x0e, 0x9f, 0x00,
    0x2f, 0x80, 0x1f, 0x00, 0x6f, 0x08, 0x27, 0x00, 0x05, 0x00, 0x09, 0x00, 0xa9, 0x5f, 0x02, 0x04, 0xff,
    0x0e, 0x9f, 0x00, 0x2f, 0x80, 0x1f, 0x00, 0x6f, 0x08, 0xc4, 0x00, 0x05, 0x00, 0x09, 0x00, 0x3d, 0x11,
    0x01, 0x04, 0xff, 0x09, 0x9f, 0x00, 0x2f, 0x80, 0x1f, 0x00, 0x9f, 0x05, 0x76, 0x00, 0x02, 0x00, 0x04,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x33, 0x90,
};

// Un EDID minimal suivi d'une extension DisplayID 2.0 dont le bloc 0x21 dit AMOLED.
std::vector<std::uint8_t> oled_edid() {
  std::vector<std::uint8_t> edid(256, 0);
  edid[0] = 0x00;
  edid[1] = 0xFF;
  edid[128] = 0x70;  // DisplayID
  edid[129] = 0x20;  // version 2.0
  edid[130] = 3 + 29;
  std::vector<std::uint8_t> block(3 + 29, 0);
  block[0] = 0x21;
  block[2] = 29;
  block[0x1E] = 0b0'010'0'101;  // technologie AMOLED (bits 6:4), profondeur native (bits 2:0)
  block[0x1C] = 0x66;           // noir 0,0001 nit en demi-flottant (0x068E)
  block[0x1D] = 0x06;
  for (std::size_t i = 0; i < block.size(); ++i) edid[133 + i] = block[i];
  return edid;
}

}  // namespace

TEST_CASE("EDID du QD-OLED de référence : modèle connu, OLED", "[app][display]") {
  ajp::app::play::FrenchUi french;
  const DisplayFacts d = parse_edid(kQdOledEdid);
  CHECK(d.vrr_min_hz == 100);  // descripteur 0xFD : 100 à 165 Hz
  // Le bloc AMD FreeSync de son CTA dit 100 à 165 Hz.
  CHECK(d.vrr_range == DisplayFacts::VrrRange{.min_hz = 100, .max_hz = 165});
  CHECK(d.vrr_floor_hz() == 100);
  CHECK(d.name == "MAG 272U E16");
  CHECK(d.manufacturer == "CND");
  CHECK(d.product == 0x8CD7);
  CHECK(d.pq);
  REQUIRE(d.peak_nits);
  CHECK(*d.peak_nits == Approx(1037.0).epsilon(0.01));
  REQUIRE(d.frame_average_nits);
  CHECK(*d.frame_average_nits == Approx(265.0).epsilon(0.01));
  REQUIRE(d.min_code);
  CHECK(*d.min_code == 2);
  CHECK_FALSE(d.displayid_oled);
  CHECK(d.source == OledSource::KnownModel);
  CHECK(d.panel == "QD-OLED");
  CHECK(d.oled);
  CHECK(describe(d).find("QD-OLED (modèle connu)") != std::string::npos);
}

TEST_CASE("Heuristique : noir code ≤ 2 et limiteur plein écran donnent OLED, sinon non", "[app][display]") {
  ajp::app::play::FrenchUi french;
  std::array<std::uint8_t, 384> edid = kQdOledEdid;
  // Un autre modèle (nom et product id changés) avec les mêmes mesures : l'heuristique seule.
  edid[10] = 0x00;
  for (std::size_t i = 54; i + 18 <= 126; i += 18) {
    if (edid[i + 3] == 0xFC) edid[i + 5] = 'X';
  }
  const DisplayFacts same = parse_edid(edid);
  CHECK(same.source == OledSource::Heuristic);
  CHECK(same.oled);
  // Noir code 10 : pas OLED.
  for (std::size_t i = 128; i + 6 < 256; ++i) {
    if (edid[i] == 0xE6 && edid[i + 1] == 0x06) edid[i + 6] = 10;
  }
  CHECK_FALSE(parse_edid(edid).oled);
}

TEST_CASE("DisplayID 2.0, bloc 0x21 : AMOLED est définitif", "[app][display]") {
  ajp::app::play::FrenchUi french;
  const DisplayFacts d = parse_edid(oled_edid());
  REQUIRE(d.displayid_oled);
  CHECK(*d.displayid_oled);
  CHECK(d.oled);
  CHECK(d.source == OledSource::DisplayId);
  REQUIRE(d.min_nits);
  CHECK(*d.min_nits < 0.001);
}

TEST_CASE("EDID trop court : OLED inconnu", "[app][display]") {
  ajp::app::play::FrenchUi french;
  CHECK(parse_edid(std::span<const std::uint8_t>(kQdOledEdid).first(64)).source == OledSource::Unknown);
}

TEST_CASE("EDID : primaires des octets 0x19 à 0x22", "[app][display]") {
  const DisplayFacts d = parse_edid(kQdOledEdid);
  REQUIRE(d.primaries);
  CHECK_THAT(d.primaries->rx, WithinAbs(0.6836, 1e-4));
  CHECK_THAT(d.primaries->ry, WithinAbs(0.3047, 1e-4));
  CHECK_THAT(d.primaries->gx, WithinAbs(0.2500, 1e-4));
  CHECK_THAT(d.primaries->gy, WithinAbs(0.7148, 1e-4));
  CHECK_THAT(d.primaries->bx, WithinAbs(0.1475, 1e-4));
  CHECK_THAT(d.primaries->by, WithinAbs(0.0615, 1e-4));
  CHECK_THAT(d.primaries->wx, WithinAbs(0.3135, 1e-4));
  CHECK_THAT(d.primaries->wy, WithinAbs(0.3291, 1e-4));
}

TEST_CASE("Contrat d'écran : KWin, puis la marge de SDL, puis l'EDID", "[app][display]") {
  const DisplayFacts d = parse_edid(kQdOledEdid);
  // KDE assombri : référence 83, pas de pic réglé, marge SDL 4 → pic 332.
  const auto dim = contract_of(d, KwinOutput{.reference = 83.0}, 4.0, std::nullopt);
  CHECK(dim.reference_nits == 83.0);
  CHECK(dim.from_compositor);
  CHECK_THAT(dim.panel_peak_nits, WithinAbs(332.0, 1e-9));
  CHECK(dim.oled);
  CHECK(dim.primaries == d.primaries);
  for (const double nits : dim.abl) CHECK(nits <= dim.panel_peak_nits);
  // Le pic réglé dans KDE l'emporte sur la marge.
  CHECK(contract_of(d, KwinOutput{.peak_override = 900.0, .reference = 203.0}, 4.0, std::nullopt)
            .panel_peak_nits == 900.0);
  // Sans KWin ni marge : l'EDID et 203.
  const auto bare = contract_of(d, std::nullopt, 1.0, std::nullopt);
  CHECK(bare.reference_nits == 203.0);
  CHECK_FALSE(bare.from_compositor);
  CHECK(bare.panel_peak_nits == d.peak_nits.value_or(1000.0));
  // La courbe mesurée remplace celle par défaut.
  const MeasuredAbl measured{.nits = {990.0, 900.0, 600.0, 400.0, 250.0}};
  CHECK(contract_of(d, std::nullopt, 1.0, measured).abl.back() == 250.0);
  CHECK(describe(dim).find("référence 83 nits (KDE") != std::string::npos);
  CHECK(describe(dim).find("pic 332 nits (marge de SDL") != std::string::npos);
  // A10-19 : rien de lu (lavapipe sous Xvfb, écran sans EDID HDR), le journal dit des valeurs par défaut.
  const std::string defaults = describe(contract_of(DisplayFacts{}, std::nullopt, 1.0, std::nullopt));
  CHECK(defaults.find("référence 203 nits (par défaut") != std::string::npos);
  CHECK(defaults.find("pic 1000 nits (par défaut") != std::string::npos);
  CHECK(describe(bare).find("nits (EDID") != std::string::npos);
}

TEST_CASE("DisplayID 2.0 : une luminance infinie ou absurde n'est pas retenue (A11-05)", "[app][display]") {
  std::vector<std::uint8_t> edid = oled_edid();
  edid[133 + 0x1A] = 0x00;  // pic : demi-flottant 0x7C00, l'infini
  edid[133 + 0x1B] = 0x7C;
  const DisplayFacts d = parse_edid(edid);
  CHECK_FALSE(d.peak_nits.has_value());  // avant : un pic infini, et la sortie restait en SDR sous Windows
  REQUIRE(d.min_nits);                   // le noir, plausible, reste lu
  edid[133 + 0x1A] = 0xFF;               // 0x7BFF : 65 504 nits, au-delà du PQ
  edid[133 + 0x1B] = 0x7B;
  CHECK_FALSE(parse_edid(edid).peak_nits.has_value());
  edid[133 + 0x1A] = 0xD0;  // 0x63D0 : 1 000 nits
  edid[133 + 0x1B] = 0x63;
  REQUIRE(parse_edid(edid).peak_nits);
  CHECK(*parse_edid(edid).peak_nits == 1000.0);
}

TEST_CASE("KWin : l'entrée de l'écran branché passe devant celle d'un ancien écran du port (A11-04)",
          "[app][display]") {
  // Deux écrans ont connu DP-1 : l'ancien, HDR actif, pic réglé à 1500 ; celui qui y est branché, à 400.
  const std::string json = R"([{"name":"outputs","data":[
    {"connectorName":"DP-1","edidIdentifier":"SAM 29187 0 12 2022 0","highDynamicRange":true,
     "maxPeakBrightnessOverride":1500,"sdrBrightness":250},
    {"connectorName":"DP-1","edidIdentifier":"MSI 15066 16843009 30 2023 0","highDynamicRange":false,
     "maxPeakBrightnessOverride":400,"sdrBrightness":180}]}])";
  CHECK(kwin_edid_key("MSI", 0x3ADA) == "MSI 15066");
  CHECK(kwin_edid_key("", 1).empty());
  const auto mine = parse_kwin_output(json, "DP-1", "MSI 15066");
  REQUIRE(mine);
  CHECK(mine->peak_override == 400.0);
  CHECK(mine->reference == 180.0);
  // Sans clé, ou clé inconnue de KWin : la règle d'avant (HDR actif, sinon la première).
  CHECK(parse_kwin_output(json, "DP-1")->peak_override == 1500.0);
  CHECK(parse_kwin_output(json, "DP-1", "GSM 1")->peak_override == 1500.0);
  // « MSI 1506 » n'est pas « MSI 15066 ».
  CHECK(parse_kwin_output(json, "DP-1", "MSI 1506")->peak_override == 1500.0);
  CHECK_FALSE(parse_kwin_output(json, "DP-2", "MSI 15066"));
}

TEST_CASE("Contrat d'écran : la fenêtre change d'écran, le contrat suit le nouvel EDID (A11-01)",
          "[app][display]") {
  DisplayFacts oled = parse_edid(kQdOledEdid);
  oled.connector.clear();  // aucun connecteur : KWin n'est pas lu, le test ne dépend pas du poste
  DisplayFacts lcd;
  lcd.peak_nits = 600.0;
  lcd.oled = false;
  ContractWatch watch(nullptr, oled, spdlog::default_logger());
  const auto first = watch.tick();
  REQUIRE(first);
  CHECK(first->oled);
  CHECK(first->panel_peak_nits == oled.peak_nits.value_or(1000.0));
  CHECK_FALSE(watch.tick());  // rien n'a changé
  watch.set_display(lcd);
  const auto moved = watch.tick();
  REQUIRE(moved);  // avant : le même contrat, celui de l'écran d'ouverture
  CHECK_FALSE(moved->oled);
  CHECK(moved->panel_peak_nits == 600.0);
  CHECK_FALSE(moved->primaries.has_value());
  // A11-02 : le changement d'écran demande aussi de relire le pic du système, une fois.
  CHECK(watch.take_peak_change());
  CHECK_FALSE(watch.take_peak_change());
}

namespace {

// Un EDID 1.4 de base dont le descripteur 0xFD dit min à max Hz ; continuous : octet 0x18, bit 0.
std::vector<std::uint8_t> base_edid(int min_hz, int max_hz, bool continuous, std::size_t blocks = 1) {
  std::vector<std::uint8_t> edid(128 * blocks, 0);
  for (std::size_t i = 1; i < 7; ++i) edid[i] = 0xFF;
  edid[0x12] = 1;
  edid[0x13] = 4;
  edid[0x18] = continuous ? 0x01 : 0x00;
  edid[0x21] = 0x50;  // blanc D65 approché, sans importance ici
  const std::size_t at = 54;
  edid[at + 3] = 0xFD;
  edid[at + 5] = static_cast<std::uint8_t>(min_hz);
  edid[at + 6] = static_cast<std::uint8_t>(max_hz);
  edid[at + 10] = 0x01;
  return edid;
}

// Une extension CTA-861 qui ne contient que le bloc propre au fabricant `payload`.
void add_cta_vendor(std::vector<std::uint8_t>& edid, std::size_t block,
                    const std::vector<std::uint8_t>& payload) {
  const std::size_t at = 128 * block;
  edid[at] = 0x02;
  edid[at + 1] = 0x03;
  edid[at + 2] = static_cast<std::uint8_t>(4 + 1 + payload.size());
  edid[at + 4] = static_cast<std::uint8_t>((3U << 5U) | payload.size());
  for (std::size_t i = 0; i < payload.size(); ++i) edid[at + 5 + i] = payload[i];
}

}  // namespace

TEST_CASE("VRR : la capacité vient de l'EDID, jamais d'une plage fixe", "[app][display]") {
  SECTION("un écran 60 Hz fixe (56 à 76 Hz, fréquence non continue) n'est pas VRR") {
    const DisplayFacts d = parse_edid(base_edid(56, 76, false));
    CHECK_FALSE(d.vrr_range.has_value());
    CHECK(d.vrr_min_hz == 56);  // le descripteur reste lu, pour le plancher
  }
  SECTION("une plage étroite, même continue, n'est pas VRR") {
    CHECK_FALSE(parse_edid(base_edid(59, 61, true)).vrr_range.has_value());
  }
  SECTION("EDID 1.4 à fréquence continue : le descripteur 0xFD") {
    CHECK(parse_edid(base_edid(48, 144, true)).vrr_range ==
          DisplayFacts::VrrRange{.min_hz = 48, .max_hz = 144});
  }
  SECTION("le maximum au-delà de 255 Hz (décalage de l'octet 4)") {
    std::vector<std::uint8_t> edid = base_edid(48, 105, true);
    edid[54 + 4] = 0x02;
    CHECK(parse_edid(edid).vrr_range == DisplayFacts::VrrRange{.min_hz = 48, .max_hz = 360});
  }
  SECTION("bloc AMD FreeSync, sans fréquence continue") {
    std::vector<std::uint8_t> edid = base_edid(56, 76, false, 2);
    add_cta_vendor(edid, 1, {0x1A, 0x00, 0x00, 0x01, 0x01, 40, 75});
    CHECK(parse_edid(edid).vrr_range == DisplayFacts::VrrRange{.min_hz = 40, .max_hz = 75});
  }
  SECTION("HF-VSDB HDMI 2.1 : VRRmin et VRRmax sur 10 bits") {
    std::vector<std::uint8_t> edid = base_edid(24, 60, false, 2);
    // OUI C4-5D-D8, version, TMDS, deux octets de drapeaux, ALLM…, puis VRRmin 48 et VRRmax 0x120 (288).
    add_cta_vendor(
        edid, 1,
        {0xD8, 0x5D, 0xC4, 0x01, 0x78, 0x00, 0x00, 0x00, static_cast<std::uint8_t>(0x40 | 48), 0x20});
    CHECK(parse_edid(edid).vrr_range == DisplayFacts::VrrRange{.min_hz = 48, .max_hz = 288});
  }
  SECTION("DisplayID 2.0, bloc 0x2B : passe devant les autres") {
    std::vector<std::uint8_t> edid = base_edid(48, 144, true, 3);
    add_cta_vendor(edid, 1, {0x1A, 0x00, 0x00, 0x01, 0x01, 40, 75});
    const std::size_t at = 256;
    edid[at] = 0x70;
    edid[at + 1] = 0x20;
    edid[at + 2] = 3 + 6;
    edid[at + 5] = 0x2B;
    edid[at + 7] = 6;
    edid[at + 8 + 2] = 30;    // minimum
    edid[at + 8 + 3] = 0x3F;  // maximum − 1 = 0x13F : 320 Hz
    edid[at + 8 + 4] = 0x01;
    CHECK(parse_edid(edid).vrr_range == DisplayFacts::VrrRange{.min_hz = 30, .max_hz = 320});
  }
}

TEST_CASE("Écran de la fenêtre : deux écrans du même modèle ne se confondent plus (A11-03)", "[app][edid]") {
  using ajp::app::play::DisplayFacts;
  using ajp::app::play::pick_display;
  DisplayFacts left;
  left.connector = "card1-DP-1";
  left.name = "MAG 272U";
  left.peak_nits = 1000.0;
  left.vrr_armed = true;
  DisplayFacts right = left;
  right.connector = "card1-DP-2";
  right.vrr_armed = false;
  DisplayFacts tv;
  tv.connector = "card1-HDMI-A-1";
  tv.name = "LG TV";
  tv.peak_nits = 800.0;
  const std::vector<DisplayFacts> found{left, right, tv};

  // Le nom du modèle seul : avant, le premier (DP-1) gagnait, avec son VRR armé, même pour la fenêtre sur
  // DP-2. L'EDID commun reste, la sortie est inconnue.
  const auto same = pick_display(found, "MAG 272U 27\"");
  REQUIRE(same);
  CHECK(same->peak_nits == 1000.0);
  CHECK(same->connector.empty());
  CHECK_FALSE(same->vrr_armed);
  // Le connecteur nommé tranche.
  CHECK(pick_display(found, "DP-2")->connector == "card1-DP-2");
  CHECK(pick_display(found, "MAG 272U (DP-2)")->vrr_armed == false);
  CHECK(pick_display(found, "DP-1")->connector == "card1-DP-1");
  // « DP-1 » ne se lit pas dans « DP-10 » : aucun nom reconnu, le premier comme avant.
  CHECK(pick_display({left, tv}, "DP-10")->connector == "card1-DP-1");
  // Un modèle unique : son connecteur et son état, comme avant.
  CHECK(pick_display(found, "LG TV")->connector == "card1-HDMI-A-1");
  CHECK(pick_display({tv}, "n'importe quoi")->connector == "card1-HDMI-A-1");
  CHECK_FALSE(pick_display({}, "DP-1"));
}

TEST_CASE("Contrat d'écran : sous Windows le pic suit le blanc SDR du système, pas 203 (A11-06)",
          "[app][edid]") {
  using ajp::app::play::contract_of;
  using ajp::app::play::DisplayFacts;
  using ajp::app::play::sdr_white_nits_of;
  // Écran de 800 nits, curseur SDR de Windows à 400 nits : SDL dit une marge de 2 et un blanc de 5,0
  // (400 / 80). Avant : 2 × 203 = 406 nits, la moitié du pic de l'écran.
  const double white = sdr_white_nits_of(5.0F, true);
  CHECK(white == 400.0);
  CHECK(contract_of(DisplayFacts{}, std::nullopt, 2.0, std::nullopt, white).panel_peak_nits == 800.0);
  CHECK(contract_of(DisplayFacts{}, std::nullopt, 2.0, std::nullopt).panel_peak_nits == 406.0);
  // Hors de Windows la propriété n'a pas ce sens : inconnue, la référence du contenu comme avant.
  CHECK(sdr_white_nits_of(5.0F, false) == 0.0);
  CHECK(sdr_white_nits_of(0.0F, true) == 0.0);
  // KDE garde la main : sa référence et son pic passent devant.
  const auto kde =
      contract_of(DisplayFacts{}, ajp::app::play::KwinOutput{.reference = 100.0}, 3.0, std::nullopt, 0.0);
  CHECK(kde.panel_peak_nits == 300.0);
}

TEST_CASE("Pic de contenu sans pic calibré : la règle de chaque système (A11-12)", "[app][edid]") {
  using ajp::app::play::PeakFacts;
  using ajp::app::play::uncalibrated_peak_nits;
  // Hors Windows : la marge × 203, rien sans HDR.
  CHECK_FALSE(uncalibrated_peak_nits({.sdl_hdr = false, .headroom = 4.0}));
  CHECK(uncalibrated_peak_nits({.sdl_hdr = true, .headroom = 4.0}) == Catch::Approx(812.0));
  // Windows, HDR vu par SDL : la marge × le blanc SDR du curseur (écran à 800 nits, blanc à 400 : marge 2),
  // bornée par le bas par la crête de l'EDID.
  CHECK(
      uncalibrated_peak_nits({.windows = true, .sdl_hdr = true, .headroom = 2.0, .sdr_white_nits = 400.0}) ==
      Catch::Approx(800.0));
  CHECK(uncalibrated_peak_nits({.windows = true,
                                .sdl_hdr = true,
                                .headroom = 2.0,
                                .sdr_white_nits = 200.0,
                                .edid_peak = 1000.0}) == Catch::Approx(1000.0));
  // Windows, HDR que SDL ne voit pas (marge de 1,00 avec Vulkan) : l'EDID, sinon 600 ; rien si Windows ne
  // dit pas le HDR actif, même avec un EDID qui annonce une crête.
  CHECK(uncalibrated_peak_nits({.windows = true, .system_hdr_active = true, .edid_peak = 1000.0}) ==
        Catch::Approx(1000.0));
  CHECK(uncalibrated_peak_nits({.windows = true, .system_hdr_active = true}) == Catch::Approx(600.0));
  CHECK_FALSE(uncalibrated_peak_nits({.windows = true, .edid_peak = 1000.0}));
}

TEST_CASE("EDID à deux sources : le journal montre le pic et le noir de chaque bloc (A11-11)",
          "[app][edid]") {
  using ajp::app::play::luminance_sources;
  // Un seul bloc parle : rien à ajouter au journal, mais sa valeur est gardée à part.
  const DisplayFacts cta_only = parse_edid(kQdOledEdid);
  REQUIRE(cta_only.cta_peak_nits);
  CHECK(cta_only.cta_peak_nits == cta_only.peak_nits);
  CHECK_FALSE(cta_only.displayid_peak_nits);
  CHECK(luminance_sources(cta_only).empty());
  const DisplayFacts displayid_only = parse_edid(oled_edid());
  REQUIRE(displayid_only.displayid_min_nits);
  CHECK(displayid_only.displayid_min_nits == displayid_only.min_nits);
  CHECK(luminance_sources(displayid_only).empty());
  // Les deux parlent, CTA puis DisplayID : aujourd'hui le pic du CTA et le noir de DisplayID.
  const DisplayFacts both{.peak_nits = 1000.0,
                          .min_nits = 0.0001,
                          .cta_peak_nits = 1000.0,
                          .cta_min_nits = 0.005,
                          .displayid_peak_nits = 950.0,
                          .displayid_min_nits = 0.0001};
  const std::string line = luminance_sources(both);
  CHECK(line.contains("pic CTA 1000 et DisplayID 950"));
  CHECK(line.contains("noir CTA 0.00500 et DisplayID 0.00010"));
  CHECK(line.contains("retenus pic 1000 et noir 0.00010"));
  CHECK(line.contains("DisplayID d'abord donnerait pic 950 et noir 0.00010"));
  CHECK(ajp::app::play::describe_for_log(both).ends_with(line));
}

TEST_CASE(
    "Écran de la fenêtre : « DP-1 » ne se lit pas dans « eDP-1 », deux cartes départagées par le nom "
    "(B11-08)",
    "[app][edid]") {
  using ajp::app::play::DisplayFacts;
  using ajp::app::play::pick_display;
  DisplayFacts external;
  external.connector = "card1-DP-1";
  external.name = "MAG 272U";
  DisplayFacts laptop;
  laptop.connector = "card1-eDP-1";
  laptop.name = "Panel";
  // Portable RTX : la fenêtre sur l'écran interne. Avant, « DP-1 » se trouvait dans « eDP-1 » et le premier
  // de la liste, l'écran externe, était pris.
  CHECK(pick_display({external, laptop}, "eDP-1")->connector == "card1-eDP-1");
  CHECK(pick_display({external, laptop}, "DP-1")->connector == "card1-DP-1");
  // Deux cartes, chacune son « DP-1 » : SDL ne dit pas la carte, le nom de l'EDID tranche entre elles.
  DisplayFacts igpu;
  igpu.connector = "card0-DP-1";
  igpu.name = "LG TV";
  DisplayFacts dgpu;
  dgpu.connector = "card1-DP-1";
  dgpu.name = "MAG 272U";
  CHECK(pick_display({igpu, dgpu}, "MAG 272U (DP-1)")->connector == "card1-DP-1");
  CHECK(pick_display({igpu, dgpu}, "LG TV (DP-1)")->connector == "card0-DP-1");
}

TEST_CASE("Écran de la fenêtre sous Windows : par son périphérique, pas par son nom (B34-02, B37-08)",
          "[app][edid]") {
  using ajp::app::play::DisplayFacts;
  using ajp::app::play::monitor_instance_from_interface;
  using ajp::app::play::pick_display;
  // Le chemin d'interface d'EnumDisplayDevices et l'identifiant d'instance de SetupAPI : la même clé.
  CHECK(monitor_instance_from_interface(
            "\\\\?\\DISPLAY#GSM5B08#5&2a0b3e4c&0&UID4352#{e6f07b5f-ee97-4a90-b076-33f57bf4eaa7}") ==
        "DISPLAY\\GSM5B08\\5&2A0B3E4C&0&UID4352");
  CHECK(monitor_instance_from_interface("DISPLAY\\GSM5B08\\5&2A0B3E4C&0&UID4352") ==
        "DISPLAY\\GSM5B08\\5&2A0B3E4C&0&UID4352");
  // Deux écrans du même modèle, nom SDL générique : avant, le premier du registre, quel que soit l'écran de
  // la fenêtre (OLED et VRR de l'autre).
  DisplayFacts first;
  first.connector = "DISPLAY\\GSM5B08\\5&2A0B3E4C&0&UID4352";
  first.name = "LG ULTRAGEAR";
  first.oled = false;
  DisplayFacts second = first;
  second.connector = "DISPLAY\\GSM5B08\\5&2A0B3E4C&0&UID4353";
  second.oled = true;
  const auto picked =
      pick_display({first, second}, "Generic PnP Monitor",
                   "\\\\?\\DISPLAY#GSM5B08#5&2a0b3e4c&0&UID4353#{e6f07b5f-ee97-4a90-b076-33f57bf4eaa7}");
  REQUIRE(picked);
  CHECK(picked->oled);
  CHECK(picked->connector == second.connector);
  // Un chemin inconnu (écran virtuel) : la règle des noms d'avant.
  CHECK(pick_display({first, second}, "LG ULTRAGEAR", "\\\\?\\DISPLAY#XXX0000#1#{x}")->connector.empty());
}

TEST_CASE("DisplayID 1.x : ses étiquettes ne se lisent pas comme celles de la 2.0 (B11-11)",
          "[app][display]") {
  using ajp::app::play::OledSource;
  using ajp::app::play::parse_edid;
  // Le même bloc 0x21 « AMOLED » que oled_edid(), dans une section qui se dit DisplayID 1.3.
  std::vector<std::uint8_t> edid = oled_edid();
  edid[129] = 0x13;
  const auto d = parse_edid(edid);
  CHECK_FALSE(d.displayid_oled);
  CHECK(d.source != OledSource::DisplayId);
  CHECK(parse_edid(oled_edid()).source == OledSource::DisplayId);  // 2.0 : lu comme avant
}

TEST_CASE("VRR armé lu dans DRM : propriété absente = inconnu, jamais « non armé » (B11-01)",
          "[app][display]") {
  using ajp::app::play::vrr_armed_from;
  // Avant : VRR_ENABLED, propriété atomique, manquait à un descripteur sans DRM_CLIENT_CAP_ATOMIC, et se
  // lisait 0, donc « non armé » en plein écran sous KDE avec le VRR engagé.
  CHECK_FALSE(vrr_armed_from(42, std::nullopt).has_value());
  CHECK(vrr_armed_from(42, 1) == true);
  CHECK(vrr_armed_from(42, 0) == false);
  CHECK(vrr_armed_from(0, std::nullopt) == false);  // sortie sans CRTC : rien n'est armé
}

TEST_CASE("Contrat d'écran sous Windows : le pic du grade (DXGI) passe devant marge × blanc (B11-10)",
          "[app][edid]") {
  using ajp::app::play::contract_of;
  using ajp::app::play::DisplayFacts;
  using Source = ajp::render::DisplayContract::PeakSource;
  // Étalonnage HDR de Windows à 950 nits, SDL dit marge 2 × blanc 400 = 800 : une seule source, celle du
  // grade.
  const auto c = contract_of(DisplayFacts{}, std::nullopt, 2.0, std::nullopt, 400.0, 950.0);
  CHECK(c.panel_peak_nits == 950.0);
  CHECK(c.peak_source == Source::Compositor);
  CHECK(contract_of(DisplayFacts{}, std::nullopt, 2.0, std::nullopt, 400.0).panel_peak_nits == 800.0);
}

TEST_CASE("Pic automatique : un téléviseur de 3000 nits est exploité (B37-09)", "[app][edid]") {
  using ajp::app::play::auto_peak_nits;
  CHECK(auto_peak_nits(3000.0) == 3000);
  CHECK(auto_peak_nits(9000.0) == 4000);
  CHECK(auto_peak_nits(150.0) == 200);
  CHECK(auto_peak_nits(std::nullopt) == 1000);
}

TEST_CASE(
    "Écran de la fenêtre sous Wayland : le connecteur de wl_output départage deux écrans du même modèle "
    "(C34-10)",
    "[app][edid]") {
  using ajp::app::play::DisplayFacts;
  using ajp::app::play::output_at;
  using ajp::app::play::pick_display;
  using ajp::app::play::WaylandOutput;
  using ajp::app::play::with_connector;
  const std::vector<WaylandOutput> outputs{{.name = "DP-1", .x = 0, .y = 0},
                                           {.name = "DP-2", .x = 2560, .y = 0}};
  CHECK(output_at(outputs, 2560, 0) == "DP-2");
  CHECK(output_at(outputs, 100, 0).empty());
  // Deux sorties au même endroit (miroir) : aucune.
  CHECK(output_at({{.name = "DP-1", .x = 0, .y = 0}, {.name = "HDMI-A-1", .x = 0, .y = 0}}, 0, 0).empty());
  CHECK(with_connector("MAG 272U", "DP-2") == "MAG 272U (DP-2)");
  CHECK(with_connector("MAG 272U", "") == "MAG 272U");

  DisplayFacts left;
  left.connector = "card1-DP-1";
  left.name = "MAG 272U";
  left.vrr_armed = true;
  DisplayFacts right = left;
  right.connector = "card1-DP-2";
  right.vrr_armed = false;
  // Le nom de SDL seul ne départage pas ; suivi du connecteur lu sur wl_output, la fenêtre sur le second
  // écran a sa sortie et son état.
  CHECK(pick_display({left, right}, "MAG 272U").value().connector.empty());
  const auto second = pick_display({left, right}, with_connector("MAG 272U", output_at(outputs, 2560, 0)));
  REQUIRE(second);
  CHECK(second->connector == "card1-DP-2");
  CHECK(second->vrr_armed == false);
}

TEST_CASE("Carte de l'écran : le connecteur DRM nomme sa carte, l'écart se dit (C37-07)", "[app][edid]") {
  using ajp::app::play::drm_card_of;
  using ajp::app::play::screen_card_line;
  using ajp::app::play::screen_card_warns;
  using ajp::app::play::ScreenCard;
  CHECK(drm_card_of("card1-DP-2") == "card1");
  CHECK(drm_card_of("card12-eDP-1") == "card12");
  CHECK(drm_card_of("DP-2").empty());
  CHECK(drm_card_of("card-DP-2").empty());
  const ScreenCard igpu{.name = "PCI 8086:a7a0", .renders = false};
  CHECK(screen_card_line(igpu, "NVIDIA GeForce RTX 4070 Laptop GPU").contains("PCI 8086:a7a0"));
  CHECK(screen_card_line(igpu, "NVIDIA GeForce RTX 4070 Laptop GPU").contains("dGPU seul"));
  // L'écart coûte en 4K ou à 144 Hz et plus ; une même carte ne dit rien.
  CHECK(screen_card_warns(igpu, 2160, 60.0));
  CHECK(screen_card_warns(igpu, 1440, 165.0));
  CHECK_FALSE(screen_card_warns(igpu, 1080, 60.0));
  CHECK_FALSE(screen_card_warns({.name = "RTX", .renders = true}, 2160, 240.0));
}
