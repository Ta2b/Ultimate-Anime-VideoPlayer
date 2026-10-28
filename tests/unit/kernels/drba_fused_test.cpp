// Découpage de DistilDRBA fusionné (spec/drba_fused.h) : réductions, formes et tampons du graphe à 1088 ×
// 1920.
#include "kernels/spec/drba_fused.h"

#include <catch2/catch_test_macros.hpp>

namespace ajp::kernels {
namespace {

constexpr DrbaFusedPlan kNormal{.tail = 4};
constexpr DrbaFusedPlan kHalfFlow{.tail = 8};

TEST_CASE("Les découpages de DistilDRBA v2 lite sont valides, d'autres non", "[kernels][drba_fused]") {
  static_assert(drba_fused_plan_valid(kNormal));
  static_assert(drba_fused_plan_valid(kHalfFlow));
  CHECK_FALSE(drba_fused_plan_valid({.tail = 0}));
  CHECK_FALSE(drba_fused_plan_valid({.tail = 2}));
  CHECK_FALSE(drba_fused_plan_valid({.tail = 6}));
  CHECK_FALSE(drba_fused_plan_valid({.tail = 16}));
  // Les réductions des blocs : 16, 8, 4 (normal) ou 32, 16, 8 (flot à l'échelle 0,5).
  CHECK(drba_fused_reduce(kNormal, 0) == 16);
  CHECK(drba_fused_reduce(kNormal, 1) == 8);
  CHECK(drba_fused_reduce(kNormal, 2) == 4);
  CHECK(drba_fused_reduce(kHalfFlow, 0) == 32);
  CHECK(drba_fused_reduce(kHalfFlow, 2) == 8);
}

TEST_CASE("Une taille remplie est prise quand chaque sortie de bloc est entière", "[kernels][drba_fused]") {
  CHECK(drba_fused_size_valid(kNormal, 1088, 1920));
  CHECK(drba_fused_size_valid(kNormal, 576, 960));
  CHECK(drba_fused_size_valid(kNormal, 32, 32));
  CHECK_FALSE(drba_fused_size_valid(kNormal, 16, 32));
  CHECK_FALSE(drba_fused_size_valid(kNormal, 100, 128));
  CHECK_FALSE(drba_fused_size_valid(kNormal, 0, 64));
  CHECK_FALSE(drba_fused_size_valid({.tail = 3}, 64, 64));
  // Le flot à l'échelle 0,5 remplit à un multiple de 128 : 64 suffit aux noyaux, 32 non.
  CHECK(drba_fused_size_valid(kHalfFlow, 2176, 3840));
  CHECK_FALSE(drba_fused_size_valid(kHalfFlow, 96, 64));
}

TEST_CASE("Les formes des entrées et sorties de blocs sont celles du graphe à 1088 × 1920",
          "[kernels][drba_fused]") {
  // X0 58 × 68 × 120, X1 109 × 136 × 240, X2 109 × 272 × 480 ; Y0 52 × 34 × 60, Y1 52 × 68 × 120, Y2 52 × 136
  // × 240.
  const auto x = [](int k) { return drba_fused_x(kNormal, k, 1088, 1920); };
  const auto y = [](int k) { return drba_fused_y(kNormal, k, 1088, 1920); };
  CHECK(x(0).channels == 58);
  CHECK(x(0).height == 68);
  CHECK(x(0).width == 120);
  CHECK(x(1).channels == 109);
  CHECK(x(1).height == 136);
  CHECK(x(2).channels == 109);
  CHECK(x(2).height == 272);
  CHECK(x(2).width == 480);
  CHECK(y(0).channels == 52);
  CHECK(y(0).height == 34);
  CHECK(y(0).width == 60);
  CHECK(y(1).height == 68);
  CHECK(y(2).height == 136);
  CHECK(y(2).width == 240);
  // En 4K au demi-flot (2176 × 3840), les blocs voient les mêmes tailles qu'en 1088 × 1920.
  CHECK(drba_fused_x(kHalfFlow, 2, 2176, 3840).height == 272);
  CHECK(drba_fused_x(kHalfFlow, 0, 2176, 3840).width == 120);
  CHECK(drba_fused_y(kHalfFlow, 2, 2176, 3840).width == 240);
}

TEST_CASE("La disposition des tampons : alignée, sans recouvrement, dans l'ordre", "[kernels][drba_fused]") {
  const DrbaFusedLayout layout = drba_fused_layout(kNormal, 1088, 1920);
  std::size_t end = 0;
  for (int k = 0; k < kDrbaFusedBlocks; ++k) {
    const auto i = static_cast<std::size_t>(k);
    CHECK(layout.x[i] == end);
    CHECK(layout.x[i] % 256 == 0);
    end += drba_fused_align(static_cast<std::size_t>(drba_fused_x(kNormal, k, 1088, 1920).elements()) * 2);
    CHECK(layout.y[i] == end);
    CHECK(layout.y[i] % 256 == 0);
    end += drba_fused_align(static_cast<std::size_t>(drba_fused_y(kNormal, k, 1088, 1920).elements()) * 2);
  }
  CHECK(layout.bytes == end);
  // Environ 41 Mo à 1088 × 1920 (X2 : 28,5 Mo), pas plus.
  CHECK(layout.bytes < std::size_t{40} << 20U);
  CHECK(layout.bytes > std::size_t{30} << 20U);
  // L'encodeur lit et écrit dans X2 : ses 9 + 48 plans sont des multiples de 16 octets (TensorRT l'exige).
  const std::size_t plane = std::size_t{272} * 480;
  CHECK((kDrbaFusedImagePlanes * plane * 2) % 16 == 0);
  CHECK((std::size_t{kDrbaFusedImagePlanes + kDrbaFusedEncodePlanes} <= std::size_t{kDrbaFusedNextChannels}));
}

}  // namespace
}  // namespace ajp::kernels
