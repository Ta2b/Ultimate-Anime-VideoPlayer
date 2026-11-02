// Le grade sur le GPU : les hooks embarqués rendus par libplacebo sur des rampes PQ BT.2020 (ou
// SDR BT.709 gamma 2,2) synthétiques, cible identique à la source (aucune conversion, aucun tramage). Un
// Vulkan logiciel (lavapipe) suffit, ce qui les fait tourner en CI ; sans aucun appareil Vulkan, ils se
// sautent. L'instance porte la couche de validation selon tests/support/vulkan_validation.h : ses
// erreurs, que libplacebo journalise en « vk … », sont notées et font échouer l'exécutable.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <libplacebo/log.h>
#include <libplacebo/renderer.h>
#include <libplacebo/vulkan.h>

#include "grade_gpu_support.h"
#include "render/cube_lut.h"
#include "render/grade.h"

namespace ajp::render {

using namespace gpu_test;

// Un orange vif (Y ≈ 0,67 R) entre dans la zone des éclats : son canal R dépasse le pic bien avant
// sa luminance.
TEST_CASE("garde-fou : un orange saturé garde son dégradé au lieu d'un plateau au pic", "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const std::vector<std::uint16_t> orange = ramp(900.0, 0.6, 0.03);
  const std::vector<double> hard = graded(gpu, orange, strong(0.0));
  const std::vector<double> soft = graded(gpu, orange, strong(1.0));
  const auto at_peak = [](const std::vector<double>& v) {
    return std::ranges::count_if(v, [](double n) { return n > 0.995 * kPeak; });
  };
  // Sans garde-fou, le haut de la rampe était écrêté au pic (canal R à 1000 nits) ; une couleur saturée sur
  // un fond clair (la rampe entière l'est) suit désormais d'elle-même l'épaule douce du traitement local.
  CHECK(at_peak(hard) == 0);
  for (std::size_t x = kSide / 2; x < hard.size(); ++x) CHECK(hard[x] >= hard[x - 1] * 0.994);
  // Avec : sous le pic, et toujours croissant (le dégradé reste).
  CHECK(at_peak(soft) == 0);
  // Croissant au pas près des textures intermédiaires de libplacebo (demi-flottants : ~0,4 % en PQ).
  for (std::size_t x = kSide / 2; x < soft.size(); ++x) CHECK(soft[x] >= soft[x - 1] * 0.994);
  CHECK(soft.back() > soft[kSide / 2] + 50.0);
}

TEST_CASE("garde-fou : le gris et le blanc ne bougent pas", "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const std::vector<std::uint16_t> gray = ramp(900.0, 1.0, 1.0);
  const std::vector<double> hard = graded(gpu, gray, strong(0.0));
  const std::vector<double> soft = graded(gpu, gray, strong(1.0));
  for (std::size_t x = 0; x < hard.size(); ++x) CHECK(std::abs(hard[x] - soft[x]) < 0.5);
}

// Le rouge pur BT.709 dans le conteneur BT.2020 : (0,6274 ; 0,0691 ; 0,0164) × R. Relu avec les
// primaires P3, il devient (0,7538 ; 0,0457 ; -0,0012) : G/R passe de 0,110 à 0,061.
TEST_CASE("couleurs étendues : un rouge BT.709 va vers le rouge P3, le gris ne bouge pas", "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const std::vector<std::uint16_t> red = ramp(400.0, 0.0691 / 0.6274, 0.0164 / 0.6274);
  const std::vector<Rgb> plain = graded_rgb(gpu, red, expanded(0.0));
  const std::vector<Rgb> wide = graded_rgb(gpu, red, expanded(1.0));
  const std::size_t mid = kSide / 2;
  const double before = plain[mid][1] / plain[mid][0];
  const double after = wide[mid][1] / wide[mid][0];
  CHECK(before > 0.10);
  CHECK(after < 0.075);
  CHECK(wide[mid][0] > plain[mid][0]);  // le canal rouge monte : la couleur est plus pure
  const std::vector<std::uint16_t> gray = ramp(400.0, 1.0, 1.0);
  const std::vector<Rgb> gray_plain = graded_rgb(gpu, gray, expanded(0.0));
  const std::vector<Rgb> gray_wide = graded_rgb(gpu, gray, expanded(1.0));
  for (std::size_t x = 0; x < gray_plain.size(); ++x)
    for (std::size_t c = 0; c < 3; ++c) CHECK(std::abs(gray_wide[x][c] - gray_plain[x][c]) < 0.5);
}

TEST_CASE("look : quinze LUT s'enchaînent, la dernière compte comme la première", "[render][gpu][lut]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const std::vector<std::uint16_t> source = sdr_ramp(1.0, 0.8, 0.6);
  const std::vector<Rgb> base = graded_codes(gpu, source, sdr_neutral());
  CubeLut inverted = identity_lut(17);
  for (float& value : inverted.rgb) value = 1.0F - value;
  std::array<const CubeLut*, kLookSlots> luts{};
  luts[3] = &inverted;
  luts[14] = &inverted;
  REQUIRE(gpu.grade->set_looks(luts));
  GradeParams params = sdr_neutral();
  params.look_mix4 = 1.0;
  params.look_mix15 = 1.0;
  const std::vector<Rgb> twice = graded_codes(gpu, source, params);
  for (std::size_t x = 8; x < twice.size(); ++x)
    for (std::size_t c = 0; c < 3; ++c) CHECK(std::abs(twice[x][c] - base[x][c]) < 8e-3);
  params.look_mix15 = 0.0;
  const std::vector<Rgb> once = graded_codes(gpu, source, params);
  CHECK(once[32][0] > 0.55);
}

TEST_CASE("HDR adaptatif : la nuit garde ses ombres, une scène claire s'étend moins", "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const auto center = [&](double nits, double adapt) {
    return graded(gpu, uniform(nits), native_look(adapt))[kSide / 2];
  };
  // Nuit (3 nits, 1,5 % du blanc SDR) : moitié du contraste sous le pivot, plus clair qu'en fixe.
  const double night_fixed = center(3.0, 0.0);
  const double night_adapt = center(3.0, 1.0);
  CHECK(night_adapt > 1.2 * night_fixed);
  // Scène claire (180 nits, 89 % du blanc, tout en aplat clair) : l'expansion est freinée.
  const double day_fixed = center(180.0, 0.0);
  const double day_adapt = center(180.0, 1.0);
  CHECK(day_fixed > 450.0);
  CHECK(day_adapt < 0.7 * day_fixed);
  CHECK(day_adapt > 180.0);
  // Scène ordinaire (60 nits) : la courbe fixe, à l'arrondi près.
  CHECK(std::abs(center(60.0, 1.0) - center(60.0, 0.0)) < 0.02 * center(60.0, 0.0));
}

TEST_CASE("sortie SDR : le grade neutre rend l'image telle quelle", "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  for (const auto& [green, blue] : {std::pair{1.0, 1.0}, std::pair{0.6, 0.1}, std::pair{0.2, 0.9}}) {
    const std::vector<std::uint16_t> source = sdr_ramp(1.0, green, blue);
    const std::vector<Rgb> out = graded_codes(gpu, source, sdr_neutral());
    const std::size_t row = static_cast<std::size_t>(kSide / 2) * kSide;
    for (std::size_t x = 0; x < out.size(); ++x)
      for (std::size_t c = 0; c < 3; ++c)
        CHECK(std::abs(out[x][c] - source[(row + x) * 4 + c] / 65535.0) < 4e-3);
  }
}

TEST_CASE("sortie SDR : couleurs et contraste poussés, sans dépasser le blanc", "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  GradeParams vivid = sdr_neutral();
  vivid.vibrance = 0.45;
  vivid.emulate = 1.0;
  vivid.sdr_white = 203.0;
  vivid.em_base = 200.0;
  vivid.em_lo = 1.19;
  vivid.em_hi = 1.14;
  vivid.em_sat = 1.1;
  // Un bleu ciel moyen : plus saturé, toujours dans le gamut BT.709.
  const std::vector<std::uint16_t> sky = sdr_ramp(0.8, 1.6, 2.0);  // codes écrêtés à 1 : B sature
  const std::vector<Rgb> plain = graded_codes(gpu, sdr_ramp(0.5, 1.4, 1.8), sdr_neutral());
  const std::vector<Rgb> pushed = graded_codes(gpu, sdr_ramp(0.5, 1.4, 1.8), vivid);
  const std::size_t mid = 3 * kSide / 4;
  CHECK(saturation(pushed[mid]) > saturation(plain[mid]) + 0.01);
  for (const Rgb& px : graded_codes(gpu, sky, vivid))
    for (const double c : px) CHECK((c >= 0.0 && c <= 1.0));
  // Le gris : le blanc reste blanc et neutre.
  const std::vector<Rgb> gray = graded_codes(gpu, sdr_ramp(1.0, 1.0, 1.0), vivid);
  CHECK(gray.back()[0] > 0.985);
  CHECK(std::abs(gray[kSide / 8][0] - gray[kSide / 8][2]) < 4e-3);
}

TEST_CASE("sortie SDR : le contraste maximal de RTX HDR n'écrase pas les sombres", "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  // Contraste 200, gris moyen 51 (les réglages de RTX HDR de Tatoub), le reste neutre.
  GradeParams look = sdr_neutral();
  look.emulate = 1.0;
  look.sdr_white = 203.0;
  look.em_base = 204.0;
  look.em_pow = 6.0 * std::pow(50.0 / 51.0, 0.35);
  look.em_lo = 1.38;
  look.em_hi = 1.28;
  look.em_sat = 1.0;
  const std::vector<Rgb> out = graded_codes(gpu, sdr_ramp(1.0, 1.0, 1.0), look);
  // La médiane d'une scène de nuit de Re:Zero : code 0,145, 2,9 nits. Avec le seul contraste sous
  // le pivot, elle tombait à 1 nit ; l'aspect TrueHDR la garde au-dessus de 2 nits.
  const auto code = [](std::size_t x) { return static_cast<double>(x + 1) / kSide; };
  for (std::size_t x = 0; x < out.size(); ++x) {
    const double in = code(x);
    if (in < 0.1 || in > 0.3) continue;
    const double nits_in = 203.0 * std::pow(in, 2.2);
    const double nits_out = 203.0 * std::pow(out[x][1], 2.2);
    CHECK(nits_out > 0.7 * nits_in);
  }
}

TEST_CASE("sortie SDR : le vrai noir OLED met les quasi-noirs à 0", "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  GradeParams black = sdr_neutral();
  // Codes 0,02 et 0,06 en gamma 2,2 à 203 nits : 0,04 et 0,4 nit environ.
  black.black_lo = 203.0 * std::pow(0.02, 2.2);
  black.black_hi = 203.0 * std::pow(0.06, 2.2);
  const std::vector<Rgb> out = graded_codes(gpu, sdr_ramp(0.25, 1.0, 1.0), black);
  // Les codes de la rampe : 0,25 × (x + 1) / 256.
  const auto code = [](std::size_t x) { return 0.25 * static_cast<double>(x + 1) / kSide; };
  for (std::size_t x = 0; x < out.size(); ++x) {
    if (code(x) < 0.015) CHECK(out[x][0] < 1e-3);
    if (code(x) > 0.09) CHECK(std::abs(out[x][0] - code(x)) < 4e-3);
  }
}

// Le look (LUT 3D) : ni la passe sautée (look_mix = 0), ni une identité ne changent l'image ; une LUT
// inversée l'inverse ; une LUT d'HDR lit les codes PQ.
TEST_CASE("look : identité et intensité nulle laissent l'image, une LUT inversée l'inverse",
          "[render][gpu][lut]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const std::vector<std::uint16_t> source = sdr_ramp(1.0, 0.8, 0.6);
  const std::vector<Rgb> base = graded_codes(gpu, source, sdr_neutral());
  const CubeLut neutral = identity_lut(17);
  REQUIRE(gpu.grade->set_looks({&neutral, nullptr, nullptr}));
  GradeParams params = sdr_neutral();
  params.look_mix = 1.0;
  params.look_white = 203.0;
  const std::vector<Rgb> same = graded_codes(gpu, source, params);
  for (std::size_t x = 8; x < same.size(); ++x)
    for (std::size_t c = 0; c < 3; ++c) CHECK(std::abs(same[x][c] - base[x][c]) < 6e-3);
  CubeLut inverted = identity_lut(17);
  for (float& value : inverted.rgb) value = 1.0F - value;
  REQUIRE(gpu.grade->set_looks({&inverted, nullptr, nullptr}));
  const std::vector<Rgb> flipped = graded_codes(gpu, source, params);
  CHECK(flipped[32][0] > 0.55);   // sombre -> clair
  CHECK(flipped[240][0] < 0.35);  // clair -> sombre
  params.look_mix = 0.0;
  const std::vector<Rgb> skipped = graded_codes(gpu, source, params);
  for (std::size_t x = 0; x < skipped.size(); ++x)
    for (std::size_t c = 0; c < 3; ++c) CHECK(skipped[x][c] == base[x][c]);  // octet pour octet
  REQUIRE(gpu.grade->set_looks({}));  // retour à l'identité 2x2x2 sans LUT
}

TEST_CASE("look : une LUT d'HDR (PQ BT.2020) lit les codes PQ, l'identité ne bouge rien",
          "[render][gpu][lut]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const std::vector<std::uint16_t> source = ramp(600.0, 0.7, 0.5);
  GradeParams neutral;
  neutral.peak = kPeak;
  neutral.shadows = 0.0;
  neutral.area_gain = 1.0;
  neutral.spec_gain = 1.0;
  neutral.vibrance = 0.0;
  neutral.gamut_p3 = 0.0;
  const std::vector<Rgb> base = graded_codes(gpu, source, neutral);
  const CubeLut lut = identity_lut(33);
  REQUIRE(gpu.grade->set_looks({&lut, nullptr, nullptr}));
  GradeParams params = neutral;
  params.look_mix = 1.0;
  params.look_pq = 1.0;
  const std::vector<Rgb> out = graded_codes(gpu, source, params);
  for (std::size_t x = 4; x < out.size(); ++x)
    for (std::size_t c = 0; c < 3; ++c) CHECK(std::abs(out[x][c] - base[x][c]) < 6e-3);
  // Le chemin SDR-dans-HDR avec une LUT neutre garde l'image à quelques millièmes près en dessous du blanc.
  params.look_pq = 0.0;
  const std::vector<Rgb> via_sdr = graded_codes(gpu, source, params);
  for (std::size_t x = 4; x < via_sdr.size(); ++x)
    for (std::size_t c = 0; c < 3; ++c) CHECK(std::abs(via_sdr[x][c] - base[x][c]) < 1.5e-2);
}

TEST_CASE("look : deux LUT s'enchaînent dans l'ordre, deux inversions se compensent", "[render][gpu][lut]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const std::vector<std::uint16_t> source = sdr_ramp(1.0, 0.8, 0.6);
  const std::vector<Rgb> base = graded_codes(gpu, source, sdr_neutral());
  CubeLut inverted = identity_lut(17);
  for (float& value : inverted.rgb) value = 1.0F - value;
  REQUIRE(gpu.grade->set_looks({&inverted, &inverted, nullptr}));
  GradeParams params = sdr_neutral();
  params.look_mix = 1.0;
  params.look_mix2 = 1.0;
  const std::vector<Rgb> twice = graded_codes(gpu, source, params);
  for (std::size_t x = 8; x < twice.size(); ++x)
    for (std::size_t c = 0; c < 3; ++c)
      CHECK(std::abs(twice[x][c] - base[x][c]) < 8e-3);  // inversion puis inversion
  params.look_mix2 = 0.0;                                // la deuxième sautée : une seule inversion
  const std::vector<Rgb> once = graded_codes(gpu, source, params);
  CHECK(once[32][0] > 0.55);
  params.look_mix = 0.0;
  params.look_mix2 = 1.0;  // seule la deuxième
  const std::vector<Rgb> second_only = graded_codes(gpu, source, params);
  CHECK(second_only[32][0] > 0.55);
  params.look_mix2 = 0.0;
  const std::vector<Rgb> none = graded_codes(gpu, source, params);
  for (std::size_t x = 0; x < none.size(); ++x)
    for (std::size_t c = 0; c < 3; ++c) CHECK(none[x][c] == base[x][c]);
}

TEST_CASE("grade adaptatif (shader) : les noirs se creusent sur une scène claire, pas sur une scène sombre",
          "[render][gpu][adaptif]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  // Un petit pan sombre (5 nits) dans une image claire (300 nits), puis la même image entièrement sombre.
  const auto split = [&](double bright_nits) {
    std::vector<std::uint16_t> pixels(static_cast<std::size_t>(kSide) * kSide * 4);
    for (int y = 0; y < kSide; ++y)
      for (int x = 0; x < kSide; ++x) {
        const double nits = x < 26 ? 5.0 : bright_nits;
        for (std::size_t c = 0; c < 3; ++c)
          pixels[(static_cast<std::size_t>(y) * kSide + static_cast<std::size_t>(x)) * 4 + c] =
              static_cast<std::uint16_t>(std::lround(nits_to_pq(nits) * 65535.0));
        pixels[(static_cast<std::size_t>(y) * kSide + static_cast<std::size_t>(x)) * 4 + 3] = 65535;
      }
    return pixels;
  };
  GradeParams params = expanded(0.0);
  params.shadows = 0.0;  // scène sombre : aucun creusement
  params.shadow_knee = 50.0;
  params.ad_on = 1.0;
  params.ad_lo = nits_to_pq(5.0);
  params.ad_hi = nits_to_pq(60.0);
  params.ad_shadows = 0.9;  // scène claire : noirs creusés
  params.ad_em_base = params.em_base;
  params.ad_em_pow = params.em_pow;
  const std::vector<Rgb> bright = graded_rgb(gpu, split(300.0), params);
  const std::vector<Rgb> dark = graded_rgb(gpu, split(8.0), params);
  params.ad_on = 0.0;
  const std::vector<Rgb> fixed = graded_rgb(gpu, split(300.0), params);
  CHECK(bright[8][0] < 2.5);  // 5 nits creusés par la scène claire
  CHECK(fixed[8][0] > 4.0);   // sans adaptation : intact
  CHECK(dark[8][0] > 4.0);    // scène sombre : intact
}

TEST_CASE("grade adaptatif (shader) : les bandes noires d'un film ne font pas passer la scène pour sombre",
          "[render][gpu][adaptif]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  // Un pan sombre (5 nits) dans une scène à 60 nits ; avec bandes noires, un quart de l'image en haut et en
  // bas.
  const auto image = [&](bool bars) {
    std::vector<std::uint16_t> pixels(static_cast<std::size_t>(kSide) * kSide * 4);
    for (int y = 0; y < kSide; ++y)
      for (int x = 0; x < kSide; ++x) {
        const bool bar = bars && (y < kSide / 4 || y >= 3 * kSide / 4);
        const double nits = bar ? 0.0 : (x < 26 ? 5.0 : 60.0);
        const std::size_t at = (static_cast<std::size_t>(y) * kSide + static_cast<std::size_t>(x)) * 4;
        for (std::size_t c = 0; c < 3; ++c)
          pixels[at + c] = static_cast<std::uint16_t>(std::lround(nits_to_pq(nits) * 65535.0));
        pixels[at + 3] = 65535;
      }
    return pixels;
  };
  GradeParams params = expanded(0.0);
  params.shadows = 0.0;
  params.shadow_knee = 50.0;
  params.ad_on = 1.0;
  params.ad_lo = nits_to_pq(10.0);
  params.ad_hi = nits_to_pq(40.0);
  params.ad_shadows = 0.9;
  params.ad_em_base = params.em_base;
  params.ad_em_pow = params.em_pow;
  const std::vector<Rgb> open = graded_rgb(gpu, image(false), params);
  const std::vector<Rgb> barred = graded_rgb(gpu, image(true), params);
  CHECK(open[8][0] < 2.5);                            // scène claire : le pan sombre est creusé
  CHECK(std::abs(barred[8][0] - open[8][0]) < 0.25);  // les bandes ne changent rien
}

TEST_CASE(
    "grade adaptatif (shader) : la calibration tient la scène claire ou l'éclat, quelle que soit l'image",
    "[render][gpu][adaptif]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  // Une image sombre (5 et 8 nits) : sans la calibration, la scène serait sombre et les noirs intacts.
  std::vector<std::uint16_t> pixels(static_cast<std::size_t>(kSide) * kSide * 4);
  for (int y = 0; y < kSide; ++y)
    for (int x = 0; x < kSide; ++x) {
      const double nits = x < 26 ? 5.0 : 8.0;
      const std::size_t at = (static_cast<std::size_t>(y) * kSide + static_cast<std::size_t>(x)) * 4;
      for (std::size_t c = 0; c < 3; ++c)
        pixels[at + c] = static_cast<std::uint16_t>(std::lround(nits_to_pq(nits) * 65535.0));
      pixels[at + 3] = 65535;
    }
  GradeParams params = expanded(0.0);
  params.shadows = 0.0;
  params.shadow_knee = 50.0;
  params.ad_on = 1.0;
  params.ad_shadows = 0.9;  // scène claire
  params.gl_shadows = 0.0;  // scène sombre avec éclat
  params.ad_em_base = params.em_base;
  params.ad_em_pow = params.em_pow;
  params.gl_em_base = params.em_base;
  params.gl_em_pow = params.em_pow;
  // Étape « scène claire » : les seuils sous la plage (grade_params.cpp, apply_adaptive).
  params.ad_lo = -2.0;
  params.ad_hi = -1.0;
  params.gl_force = 0.0;
  const std::vector<Rgb> bright = graded_rgb(gpu, pixels, params);
  // Étape « éclat » : ses noirs à elle, creusés à leur tour.
  params.ad_lo = 2.0;
  params.ad_hi = 3.0;
  params.gl_force = 1.0;
  params.gl_shadows = 0.9;
  const std::vector<Rgb> glow = graded_rgb(gpu, pixels, params);
  // Étape « scène sombre » : les curseurs de base, intacts.
  params.gl_force = 0.0;
  const std::vector<Rgb> dark = graded_rgb(gpu, pixels, params);
  CHECK(bright[8][0] < 2.5);
  CHECK(glow[8][0] < 2.5);
  CHECK(dark[8][0] > 4.0);
}

TEST_CASE("grade adaptatif (shader) : la position du plan analysé remplace la mesure du shader",
          "[render][gpu][adaptif]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  // Une image sombre : la mesure du shader la dit sombre, le plan analysé la dit claire.
  std::vector<std::uint16_t> pixels(static_cast<std::size_t>(kSide) * kSide * 4);
  for (int y = 0; y < kSide; ++y)
    for (int x = 0; x < kSide; ++x) {
      const double nits = x < 26 ? 5.0 : 8.0;
      const std::size_t at = (static_cast<std::size_t>(y) * kSide + static_cast<std::size_t>(x)) * 4;
      for (std::size_t c = 0; c < 3; ++c)
        pixels[at + c] = static_cast<std::uint16_t>(std::lround(nits_to_pq(nits) * 65535.0));
      pixels[at + 3] = 65535;
    }
  GradeParams params = expanded(0.0);
  params.shadows = 0.0;
  params.shadow_knee = 50.0;
  params.ad_on = 1.0;
  params.ad_lo = nits_to_pq(10.0);
  params.ad_hi = nits_to_pq(60.0);
  params.ad_shadows = 0.9;
  params.ad_em_base = params.em_base;
  params.ad_em_pow = params.em_pow;
  const std::vector<Rgb> measured = graded_rgb(gpu, pixels, params);
  params.sh_on = 1.0;
  params.sh_t = 1.0;
  const std::vector<Rgb> analyzed = graded_rgb(gpu, pixels, params);
  CHECK(measured[8][0] > 4.0);  // la mesure : scène sombre, noirs intacts
  CHECK(analyzed[8][0] < 2.5);  // le plan analysé : scène claire, noirs creusés
}

}  // namespace ajp::render

namespace ajp::render {

// La courbe de KWin sans métadonnées (colormanagement.glsl, doTonemapping) sur un gris : le signal (blanc
// PQ de 203 nits) posé au blanc de référence de KDE, compressé de 10 000 nits vers le pic de l'écran.
double kwin_without_metadata(double nits, double reference, double peak) {
  const double k = reference / 203.0;
  const double in_range = 10000.0 * k / reference;
  const double out_range = peak / reference;
  const double v = (out_range * (1.0 + in_range) - in_range) / (in_range * in_range);
  const double rel = nits * k / reference;
  return std::min(rel * (1.0 + rel * v) / (1.0 + rel) * reference, peak);
}

TEST_CASE("métadonnées HDR : le grade compresse comme KWin, qui n'a plus qu'à écrêter", "[render][gpu]") {
  using namespace gpu_test;
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const std::vector<std::uint16_t> source = ramp(4000.0, 1.0, 1.0);  // un gris jusqu'à 4 000 nits
  GradeParams neutral;
  neutral.peak = kPeak;
  neutral.shadows = 0.0;
  neutral.area_gain = 1.0;
  neutral.spec_gain = 1.0;
  neutral.vibrance = 0.0;
  neutral.gamut_p3 = 0.0;
  const std::vector<Rgb> base = graded_rgb(gpu, source, neutral);
  constexpr double kReference = 200.0;
  constexpr double kScreenPeak = 800.0;
  gpu.grade->set_kwin_match(Grade::KwinMatch{.reference = kReference, .peak = kScreenPeak});
  const std::vector<Rgb> matched = graded_rgb(gpu, source, neutral);
  gpu.grade->set_kwin_match(std::nullopt);
  const double k = kReference / 203.0;
  for (std::size_t x = 4; x < base.size(); ++x) {
    const double before = kwin_without_metadata(base[x][0], kReference, kScreenPeak);
    const double now = std::min(matched[x][0] * k, kScreenPeak);  // KWin, métadonnées reçues : il écrête
    CHECK(std::abs(now - before) <= (0.02 * before) + 0.05);
    CHECK(std::abs(matched[x][1] - matched[x][0]) <= 0.01 * matched[x][0] + 0.01);  // le gris reste gris
  }
  // Sans la passe, l'image ne bouge pas.
  const std::vector<Rgb> off = graded_rgb(gpu, source, neutral);
  for (std::size_t x = 0; x < base.size(); ++x) CHECK(off[x][0] == base[x][0]);
}

}  // namespace ajp::render
