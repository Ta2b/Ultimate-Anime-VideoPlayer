#include "app/play/display_edid.h"

#include <array>
#include <cstdint>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "app/play/display_contract.h"
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
  CHECK(describe(dim).find("référence 83 nits") != std::string::npos);
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
