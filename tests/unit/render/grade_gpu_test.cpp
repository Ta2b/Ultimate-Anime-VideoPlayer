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

TEST_CASE("look : les LUT s'enchaînent, la dernière compte comme la première", "[render][gpu][lut]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const std::vector<std::uint16_t> source = pq_of_sdr_ramp(1.0, 0.8, 0.6);
  const std::vector<Rgb> base = graded_codes(gpu, source, look_neutral(), true);
  CubeLut inverted = identity_lut(17);
  for (float& value : inverted.rgb) value = 1.0F - value;
  std::array<const CubeLut*, kLookSlots> luts{};
  luts[3] = &inverted;
  luts[kLookSlots - 1] = &inverted;
  REQUIRE(gpu.grade->set_looks(luts));
  GradeParams params = look_neutral();
  params.look_mixes[3] = 1.0;
  params.look_mixes[kLookSlots - 1] = 1.0;
  const std::vector<Rgb> twice = graded_codes(gpu, source, params, true);
  for (std::size_t x = 8; x < twice.size(); ++x)
    for (std::size_t c = 0; c < 3; ++c) CHECK(std::abs(twice[x][c] - base[x][c]) < 8e-3);
  params.look_mixes[kLookSlots - 1] = 0.0;
  const std::vector<Rgb> once = graded_codes(gpu, source, params, true);
  CHECK(once[32][0] > base[32][0] + 0.1);  // sombre -> clair
}

// Le look (LUT 3D) : ni la passe sautée (look_mix = 0), ni une identité ne changent l'image ; une LUT
// inversée l'inverse ; une LUT d'HDR lit les codes PQ.
TEST_CASE("look : identité et intensité nulle laissent l'image, une LUT inversée l'inverse",
          "[render][gpu][lut]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const std::vector<std::uint16_t> source = pq_of_sdr_ramp(1.0, 0.8, 0.6);
  const std::vector<Rgb> base = graded_codes(gpu, source, look_neutral(), true);
  const CubeLut neutral = identity_lut(17);
  REQUIRE(gpu.grade->set_looks({&neutral, nullptr, nullptr}));
  GradeParams params = look_neutral();
  params.look_mixes[0] = 1.0;
  const std::vector<Rgb> same = graded_codes(gpu, source, params, true);
  for (std::size_t x = 8; x < same.size(); ++x)
    for (std::size_t c = 0; c < 3; ++c) CHECK(std::abs(same[x][c] - base[x][c]) < 6e-3);
  CubeLut inverted = identity_lut(17);
  for (float& value : inverted.rgb) value = 1.0F - value;
  REQUIRE(gpu.grade->set_looks({&inverted, nullptr, nullptr}));
  const std::vector<Rgb> flipped = graded_codes(gpu, source, params, true);
  CHECK(flipped[32][0] > base[32][0] + 0.1);    // sombre -> clair
  CHECK(flipped[240][0] < base[240][0] - 0.1);  // clair -> sombre
  params.look_mixes[0] = 0.0;
  const std::vector<Rgb> skipped = graded_codes(gpu, source, params, true);
  for (std::size_t x = 0; x < skipped.size(); ++x)
    for (std::size_t c = 0; c < 3; ++c) CHECK(skipped[x][c] == base[x][c]);  // octet pour octet
  REQUIRE(
      gpu.grade->set_looks(std::array<const CubeLut*, kLookSlots>{}));  // retour à l'identité 2x2x2 sans LUT
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
  params.look_mixes[0] = 1.0;
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
  const std::vector<std::uint16_t> source = pq_of_sdr_ramp(1.0, 0.8, 0.6);
  const std::vector<Rgb> base = graded_codes(gpu, source, look_neutral(), true);
  CubeLut inverted = identity_lut(17);
  for (float& value : inverted.rgb) value = 1.0F - value;
  REQUIRE(gpu.grade->set_looks({&inverted, &inverted, nullptr}));
  GradeParams params = look_neutral();
  params.look_mixes[0] = 1.0;
  params.look_mixes[1] = 1.0;
  const std::vector<Rgb> twice = graded_codes(gpu, source, params, true);
  for (std::size_t x = 8; x < twice.size(); ++x)
    for (std::size_t c = 0; c < 3; ++c)
      CHECK(std::abs(twice[x][c] - base[x][c]) < 8e-3);  // inversion puis inversion
  params.look_mixes[1] = 0.0;                            // la deuxième sautée : une seule inversion
  const std::vector<Rgb> once = graded_codes(gpu, source, params, true);
  CHECK(once[32][0] > base[32][0] + 0.1);
  params.look_mixes[0] = 0.0;
  params.look_mixes[1] = 1.0;  // seule la deuxième
  const std::vector<Rgb> second_only = graded_codes(gpu, source, params, true);
  CHECK(second_only[32][0] > base[32][0] + 0.1);
  params.look_mixes[1] = 0.0;
  const std::vector<Rgb> none = graded_codes(gpu, source, params, true);
  for (std::size_t x = 0; x < none.size(); ++x)
    for (std::size_t c = 0; c < 3; ++c) CHECK(none[x][c] == base[x][c]);
}

}  // namespace ajp::render
