// Le corpus de cas durs de la porte du SPAN FP8 (étape 3.7a) : déterministe, à la taille demandée, varié.
#include "gpu/span_fp8/hard_cases.h"

#include <algorithm>
#include <set>
#include <string>

#include <catch2/catch_test_macros.hpp>

namespace ajp::gpu::span_fp8 {

TEST_CASE("les cas durs sont toujours les mêmes, à la taille demandée", "[span_fp8][hard_cases]") {
  const std::vector<HardCase> a = hard_cases(128, 72);
  const std::vector<HardCase> b = hard_cases(128, 72);
  REQUIRE(a.size() == 9);
  std::set<std::string> names;
  for (std::size_t i = 0; i < a.size(); ++i) {
    INFO(a[i].name);
    names.insert(a[i].name);
    CHECK(a[i].width == 128);
    CHECK(a[i].height == 72);
    CHECK(a[i].rgb.size() == std::size_t{3} * 128 * 72);
    CHECK(a[i].rgb == b[i].rgb);
    // Chaque cas a du contraste : au moins deux niveaux.
    const auto [lo, hi] = std::ranges::minmax(a[i].rgb);
    CHECK(hi > lo);
  }
  CHECK(names.size() == a.size());
  CHECK(hard_cases(8, 8).empty());
  CHECK(hard_cases(127, 72).empty());
}

TEST_CASE("les rampes sombres restent sous 36 et les bords saturés aux bornes", "[span_fp8][hard_cases]") {
  const std::vector<HardCase> cases = hard_cases(64, 48);
  const auto find = [&](const std::string& name) {
    return *std::ranges::find_if(cases, [&](const HardCase& c) { return c.name == name; });
  };
  const HardCase ramps = find("rampes sombres");
  CHECK(std::ranges::max(ramps.rgb) <= 35);
  const HardCase edges = find("bords saturés");
  CHECK(std::ranges::all_of(edges.rgb, [](std::uint8_t v) { return v == 0 || v == 255; }));
}

}  // namespace ajp::gpu::span_fp8
