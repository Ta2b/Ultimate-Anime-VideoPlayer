#include "kernels/ref/gmfss.h"

#include <cmath>
#include <limits>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "kernels/spec.h"

using namespace ajp::kernels;
using Catch::Matchers::WithinAbs;

namespace {

// Un plan dont chaque échantillon vaut x + 10 y + offset.
PlaneF ramp(int w, int h, float offset) {
  PlaneF p(w, h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) p.at(x, y) = static_cast<float>(x + 10 * y) + offset;
  }
  return p;
}

ref::SplatMotion uniform(int w, int h, float fx, float fy, float z) {
  return {.planes = {PlaneF(w, h, fx), PlaneF(w, h, fy), PlaneF(w, h, z)}};
}

}  // namespace

TEST_CASE("Sans mouvement, le splat rend la source", "[kernels][gmfss]") {
  const std::vector<PlaneF> source{ramp(6, 4, 0.0F), ramp(6, 4, 0.5F)};
  for (const float s : {0.0F, 0.5F, 1.0F}) {
    auto out = ref::soft_splat(source, uniform(6, 4, 0.0F, 0.0F, 3.0F), s);
    REQUIRE(out);
    REQUIRE(out->size() == 2);
    for (std::size_t c = 0; c < 2; ++c) {
      for (std::size_t i = 0; i < source[c].data.size(); ++i)
        CHECK_THAT((*out)[c].data[i], WithinAbs(source[c].data[i], 1e-4));
    }
  }
}

TEST_CASE("Un flot entier déplace la source, l'instant le met à l'échelle", "[kernels][gmfss]") {
  const std::vector<PlaneF> source{ramp(8, 3, 1.0F)};
  // Flot de 4 pixels vers la droite, à s = 1/2 : 2 pixels.
  auto out = ref::soft_splat(source, uniform(8, 3, 4.0F, 0.0F, 0.0F), 0.5F);
  REQUIRE(out);
  for (int y = 0; y < 3; ++y) {
    for (int x = 0; x < 8; ++x) {
      const float expected = x >= 2 ? source[0].at(x - 2, y) : 0.0F;  // rien n'atteint les deux premiers
      CHECK_THAT((*out)[0].at(x, y), WithinAbs(expected, 1e-4));
    }
  }
}

TEST_CASE("Deux sources sur un pixel se mêlent par le softmax de leur métrique", "[kernels][gmfss]") {
  // Pixel 0 (valeur 2) reste ; pixel 1 (valeur 6) recule d'un pixel : les deux tombent en 0.
  PlaneF value(2, 1);
  value.at(0, 0) = 2.0F;
  value.at(1, 0) = 6.0F;
  ref::SplatMotion motion = uniform(2, 1, 0.0F, 0.0F, 0.0F);
  motion.planes[0].at(1, 0) = -1.0F;
  motion.planes[2].at(0, 0) = 1.0F;
  motion.planes[2].at(1, 0) = -1.0F;
  auto out = ref::soft_splat({value}, motion, 1.0F);
  REQUIRE(out);
  const double e1 = std::exp(1.0);
  const double em1 = std::exp(-1.0);
  CHECK_THAT((*out)[0].at(0, 0), WithinAbs((2.0 * e1 + 6.0 * em1) / (e1 + em1), 1e-5));
  CHECK_THAT((*out)[0].at(1, 0), WithinAbs(0.0, 1e-6));
}

TEST_CASE("Un demi-pixel partage la source entre deux voisins", "[kernels][gmfss]") {
  PlaneF value(4, 1, 0.0F);
  value.at(1, 0) = 8.0F;
  ref::SplatMotion motion = uniform(4, 1, 0.0F, 0.0F, 0.0F);
  motion.planes[0].at(1, 0) = 0.5F;
  // Les pixels 0, 2 et 3 restent en place, valeur 0 : le pixel 1 reçoit la moitié de 8 seule (son
  // poids vaut 1/2) et le pixel 2 la moitié de 8 avec tout le 0 du pixel 2.
  auto out = ref::soft_splat({value}, motion, 1.0F);
  REQUIRE(out);
  CHECK_THAT((*out)[0].at(1, 0), WithinAbs(8.0, 1e-5));
  CHECK_THAT((*out)[0].at(2, 0), WithinAbs(4.0 / 1.5, 1e-5));
}

TEST_CASE("Un flot non fini n'apporte rien", "[kernels][gmfss]") {
  const std::vector<PlaneF> source{PlaneF(3, 1, 5.0F)};
  ref::SplatMotion motion = uniform(3, 1, 0.0F, 0.0F, 0.0F);
  motion.planes[0].at(1, 0) = std::numeric_limits<float>::quiet_NaN();
  auto out = ref::soft_splat(source, motion, 1.0F);
  REQUIRE(out);
  CHECK_THAT((*out)[0].at(0, 0), WithinAbs(5.0, 1e-5));
  CHECK((*out)[0].at(1, 0) == 0.0F);
}

TEST_CASE("Le splat refuse ce qui sort du contrat", "[kernels][gmfss]") {
  const std::vector<PlaneF> source{PlaneF(4, 4)};
  CHECK(ref::soft_splat({}, uniform(4, 4, 0, 0, 0), 0.5F).error().code == ErrorCode::SizeMismatch);
  CHECK(ref::soft_splat(source, uniform(4, 3, 0, 0, 0), 0.5F).error().code == ErrorCode::SizeMismatch);
  CHECK(ref::soft_splat(source, uniform(4, 4, 0, 0, 0), 1.5F).error().code == ErrorCode::ParameterOutOfRange);
  CHECK(gmfss_side_time(0, 0.25F) == 0.25F);
  CHECK(gmfss_side_time(1, 0.25F) == 0.75F);
}
