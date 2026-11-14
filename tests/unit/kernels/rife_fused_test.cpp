// Découpage de RIFE fusionné (spec/rife_fused.h) et sa référence CPU (ref/rife_fused.h).
#include "kernels/ref/rife_fused.h"

#include <array>
#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "kernels/spec/rife.h"
#include "kernels/spec/rife_fused.h"

namespace ajp::kernels {
namespace {

constexpr RifeFusedPlan kV425{.blocks = 5, .reduce = {8, 8, 4, 2, 1}};
constexpr RifeFusedPlan kV422{.blocks = 4, .reduce = {8, 4, 2, 1}};

TEST_CASE("Les découpages de RIFE 4.25 et 4.22 lite sont valides, d'autres non", "[kernels][rife_fused]") {
  static_assert(rife_fused_plan_valid(kV425));
  static_assert(rife_fused_plan_valid(kV422));
  CHECK_FALSE(rife_fused_plan_valid({}));
  CHECK_FALSE(rife_fused_plan_valid({.blocks = 1, .reduce = {1}}));
  CHECK_FALSE(rife_fused_plan_valid({.blocks = 3, .reduce = {8, 4, 2}}));  // le dernier doit valoir 1
  CHECK_FALSE(
      rife_fused_plan_valid({.blocks = 3, .reduce = {1, 2, 1}}));  // le premier : une moyenne 2 × 2 au moins
  CHECK_FALSE(rife_fused_plan_valid({.blocks = 3, .reduce = {8, 3, 1}}));    // pas une puissance de deux
  CHECK(rife_fused_plan_valid({.blocks = 5, .reduce = {16, 8, 4, 2, 1}}));   // RIFE 4.26
  CHECK(rife_fused_plan_valid({.blocks = 5, .reduce = {32, 16, 8, 4, 1}}));  // RIFE 4.25 lite
  CHECK_FALSE(rife_fused_plan_valid({.blocks = 3, .reduce = {64, 4, 1}}));   // plus de 32
  RifeFusedPlan six{.blocks = 6, .reduce = {8, 8, 4, 2, 1}};
  CHECK_FALSE(rife_fused_plan_valid(six));
}

TEST_CASE("RIFE fusionné : une taille remplie est prise quand chaque sortie de bloc est entière",
          "[kernels][rife_fused]") {
  CHECK(rife_fused_size_valid(kV425, 768, 1280));
  CHECK(rife_fused_size_valid(kV425, 64, 64));
  CHECK(rife_fused_size_valid(kV422, 1088, 1920));
  CHECK_FALSE(rife_fused_size_valid(kV425, 100, 128));
  CHECK_FALSE(rife_fused_size_valid(kV425, 8, 16));  // Y0 = 8 / 16 : pas entier
  CHECK_FALSE(rife_fused_size_valid(kV425, 0, 64));
  CHECK_FALSE(rife_fused_size_valid({}, 64, 64));
  // 4.25 lite : ses convolutions du premier bloc travaillent au 1/128, d'où son remplissage à 128.
  constexpr RifeFusedPlan lite{.blocks = 5, .reduce = {32, 16, 8, 4, 1}};
  CHECK(rife_fused_size_valid(lite, 256, 384));
  CHECK_FALSE(rife_fused_size_valid(lite, 192, 384));
  CHECK_FALSE(rife_fused_size_valid({.blocks = 2, .reduce = {8, 1}}, 48, 64));  // 48 / 32 : pas entier
  // Toute taille que RIFE remplit (multiples de 64) convient.
  for (const int size : {64, 128, 192, 576, 720 + 48, 1088}) {
    CHECK(rife_fused_size_valid(kV425, size, 64));
    CHECK(rife_fused_size_valid(kV422, 64, size));
  }
}

TEST_CASE("Les formes des entrées et sorties de blocs sont celles du graphe à 768 × 1280",
          "[kernels][rife_fused]") {
  // RIFE 4.25 : X0 15 × 96 × 160, X1 28 × 96 × 160, X2 28 × 192 × 320, X3 28 × 384 × 640, X4 28 × 768 × 1280
  // ; Y0 et Y1 52 × 48 × 80, Y2 52 × 96 × 160, Y3 52 × 192 × 320, Y4 52 × 384 × 640.
  const auto x = [](int k) { return rife_fused_x(kV425, k, 768, 1280); };
  const auto y = [](int k) { return rife_fused_y(kV425, k, 768, 1280); };
  CHECK(x(0).channels == 15);
  CHECK(x(0).height == 96);
  CHECK(x(0).width == 160);
  CHECK(x(1).channels == 28);
  CHECK(x(1).height == 96);
  CHECK(x(2).height == 192);
  CHECK(x(3).width == 640);
  CHECK(x(4).height == 768);
  CHECK(x(4).width == 1280);
  CHECK(y(0).channels == 52);
  CHECK(y(0).height == 48);
  CHECK(y(1).width == 80);
  CHECK(y(2).height == 96);
  CHECK(y(3).width == 320);
  CHECK(y(4).height == 384);
  CHECK(y(4).width == 640);
  // 4.22 lite : X1 28 × 192 × 320 (réduit 4), Y3 52 × 384 × 640.
  CHECK(rife_fused_x(kV422, 1, 768, 1280).height == 192);
  CHECK(rife_fused_y(kV422, 3, 768, 1280).height == 384);
}

TEST_CASE("RIFE fusionné : la disposition des tampons, alignée, sans recouvrement, dans l'ordre",
          "[kernels][rife_fused]") {
  const RifeFusedLayout layout = rife_fused_layout(kV425, 768, 1280);
  CHECK(layout.encode == 0);
  std::size_t end = fused_align(std::size_t{8} * 768 * 1280 * 2);
  for (int k = 0; k < kV425.blocks; ++k) {
    const auto i = static_cast<std::size_t>(k);
    CHECK(layout.x[i] == end);
    CHECK(layout.x[i] % 256 == 0);
    end += fused_align(static_cast<std::size_t>(rife_fused_x(kV425, k, 768, 1280).elements()) * 2);
    CHECK(layout.y[i] == end);
    CHECK(layout.y[i] % 256 == 0);
    end += fused_align(static_cast<std::size_t>(rife_fused_y(kV425, k, 768, 1280).elements()) * 2);
  }
  CHECK(layout.bytes == end);
  // Environ 120 Mo à 768 × 1280, pas plus : la zone d'activation d'un RIFE entier en pèse mille.
  CHECK(layout.bytes < std::size_t{130} << 20U);
  CHECK(layout.bytes > std::size_t{80} << 20U);
}

TEST_CASE("RIFE fusionné : en HWC8, X et Y prennent leurs canaux arrondis au multiple de 8",
          "[kernels][rife_fused]") {
  CHECK(rife_fused_hwc8_pitch(15) == 16);
  CHECK(rife_fused_hwc8_pitch(28) == 32);
  CHECK(rife_fused_hwc8_pitch(52) == 56);
  const RifeFusedLayout planes = rife_fused_layout(kV425, 768, 1280);
  const RifeFusedLayout hwc8 = rife_fused_layout(kV425, 768, 1280, true);
  CHECK(hwc8.encode == planes.encode);
  CHECK(hwc8.x[0] == planes.x[0]);
  std::size_t end = hwc8.x[0];
  for (int k = 0; k < kV425.blocks; ++k) {
    const auto i = static_cast<std::size_t>(k);
    const FusedDims x = rife_fused_x(kV425, k, 768, 1280);
    const FusedDims y = rife_fused_y(kV425, k, 768, 1280);
    CHECK(hwc8.x[i] == end);
    const FusedDims xp{.channels = rife_fused_hwc8_pitch(x.channels), .height = x.height, .width = x.width};
    const FusedDims yp{.channels = rife_fused_hwc8_pitch(y.channels), .height = y.height, .width = y.width};
    end += fused_align(static_cast<std::size_t>(xp.elements()) * 2);
    CHECK(hwc8.y[i] == end);
    end += fused_align(static_cast<std::size_t>(yp.elements()) * 2);
  }
  CHECK(hwc8.bytes == end);
  CHECK(hwc8.bytes > planes.bytes);
}

// Sans flot (les Y sont nuls), le warp ne déplace rien : l'image A de X1 est celle de X0 quand les deux
// réduisent de la même manière, et les canaux de flot sont nuls.
TEST_CASE("La référence : un flot nul laisse les images en place", "[kernels][rife_fused]") {
  constexpr int h = 64;
  constexpr int w = 128;
  const auto area = static_cast<std::size_t>(h) * w;
  ref::RifeFusedState state{.plan = kV425, .height = h, .width = w};
  state.input.assign(11 * area, 0.0);
  for (std::size_t i = 0; i < 6 * area; ++i) state.input[i] = std::fmod(static_cast<double>(i) * 0.37, 1.0);
  for (std::size_t i = 0; i < area; ++i) state.input[6 * area + i] = 0.25;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const std::size_t i = static_cast<std::size_t>(y) * w + static_cast<std::size_t>(x);
      state.input[7 * area + i] = rife_mesh(x, w);
      state.input[8 * area + i] = rife_mesh(y, h);
      state.input[9 * area + i] = rife_mesh_step(w);
      state.input[10 * area + i] = rife_mesh_step(h);
    }
  }
  state.encode.assign(8 * area, 0.5);
  for (int k = 0; k < kV425.blocks; ++k)
    state.y.emplace_back(static_cast<std::size_t>(rife_fused_y(kV425, k, h, w).elements()), 0.0);

  const std::vector<double> x0 = ref::rife_fused_block_input(state, 0);
  const std::vector<double> x1 = ref::rife_fused_block_input(state, 1);
  REQUIRE(x0.size() == static_cast<std::size_t>(rife_fused_x(kV425, 0, h, w).elements()));
  REQUIRE(x1.size() == static_cast<std::size_t>(rife_fused_x(kV425, 1, h, w).elements()));
  const auto small = static_cast<std::size_t>(rife_fused_x(kV425, 1, h, w).height) *
                     static_cast<std::size_t>(rife_fused_x(kV425, 1, h, w).width);
  for (std::size_t c = 0; c < 6; ++c) {
    for (std::size_t i = 0; i < small; ++i)
      CHECK(x1[c * small + i] == Catch::Approx(x0[c * small + i]).margin(1e-4));
  }
  for (std::size_t i = 0; i < small; ++i) {
    CHECK(x1[14 * small + i] == Catch::Approx(0.25).margin(1e-4));  // l'instant
    for (std::size_t c = 24; c < 28; ++c) CHECK(x1[c * small + i] == 0.0);
  }
  // La fin : sans flot, le masque sigmoïde de 0 vaut 1/2 et l'image est la moyenne des deux images.
  const ref::RifeFusedResult result = ref::rife_fused_finish(state);
  for (std::size_t i = 0; i < area; ++i) {
    CHECK(result.mask[i] == Catch::Approx(0.5));
    CHECK(result.flow[i] == 0.0);
    CHECK(result.image[i] == Catch::Approx(0.5 * (state.input[i] + state.input[3 * area + i])).margin(1e-4));
  }
  // Catmull-Rom passe par les points : sans flot, la même image.
  const ref::RifeFusedResult cubic = ref::rife_fused_finish(state, RifeWarp::CatmullRom);
  for (std::size_t i = 0; i < 3 * area; ++i)
    CHECK(cubic.image[i] == Catch::Approx(result.image[i]).margin(1e-9));
}

TEST_CASE("Catmull-Rom : poids de somme 1, passe par les points, rend une rampe exacte",
          "[kernels][rife_fused]") {
  for (const double t : {0.0, 0.1, 0.25, 0.5, 0.75, 0.9}) {
    const std::array<double, 4> w = catmull_rom_weights(t);
    CHECK(w[0] + w[1] + w[2] + w[3] == Catch::Approx(1.0));
    // Une rampe p(i) = i : la cubique la rend telle quelle (elle reproduit les polynômes de degré 1).
    CHECK(-w[0] + w[2] + 2.0 * w[3] == Catch::Approx(t).margin(1e-12));
  }
  const std::array<double, 4> at0 = catmull_rom_weights(0.0);
  CHECK(at0 == std::array<double, 4>{0.0, 1.0, 0.0, 0.0});
  // Un demi-pixel entre deux points d'un bord net : le dépassement de la cubique, que la sortie borne.
  const std::array<double, 4> half = catmull_rom_weights(0.5);
  CHECK(half[0] == Catch::Approx(-0.0625));
  CHECK(half[1] == Catch::Approx(0.5625));
}

}  // namespace
}  // namespace ajp::kernels
