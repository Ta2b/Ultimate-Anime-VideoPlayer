// Le grade sur le GPU : les hooks embarqués rendus par libplacebo sur des rampes PQ BT.2020 (ou
// SDR BT.709 gamma 2,2) synthétiques, cible identique à la source (aucune conversion, aucun tramage). Un
// Vulkan logiciel (lavapipe) suffit, ce qui les fait tourner en CI ; sans aucun appareil Vulkan, ils se
// sautent. L'instance porte la couche de validation selon tests/support/vulkan_validation.h : ses
// erreurs, que libplacebo journalise en « vk … », sont notées et font échouer l'exécutable.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <libplacebo/log.h>
#include <libplacebo/renderer.h>
#include <libplacebo/vulkan.h>

#include "grade_gpu_support.h"
#include "render/cube_lut.h"
#include "render/grade.h"

namespace ajp::render {

using namespace gpu_test;

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

TEST_CASE("look : la pile composée en tâche de fond rend la même image (P4.9)", "[render][gpu][lut]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const std::vector<std::uint16_t> source = sdr_ramp(1.0, 0.8, 0.6);
  CubeLut inverted = identity_lut(17);
  for (float& value : inverted.rgb) value = 1.0F - value;
  REQUIRE(gpu.grade->set_looks({&inverted, nullptr, nullptr}));
  GradeParams params = sdr_neutral();
  params.look_mix = 1.0;
  const std::vector<Rgb> direct = graded_codes(gpu, source, params);
  gpu.grade->set_background(true);
  REQUIRE(gpu.grade->set_looks({&inverted, nullptr, nullptr}));  // la pile est à refaire, sur le thread
  gpu.grade->set_params(params);
  bool adopted = false;
  for (int i = 0; i < 500 && !adopted; ++i) {
    adopted = gpu.grade->adopt();
    if (!adopted) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  REQUIRE(adopted);
  REQUIRE(gpu.grade->look_stack() != nullptr);
  const std::vector<Rgb> later = graded_codes(gpu, source, params);
  for (std::size_t x = 8; x < later.size(); ++x)
    for (std::size_t c = 0; c < 3; ++c) CHECK(std::abs(later[x][c] - direct[x][c]) < 1e-4);
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
  neutral.spec_gain = 1.0;
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
  neutral.spec_gain = 1.0;
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
