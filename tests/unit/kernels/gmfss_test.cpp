#include "kernels/ref/gmfss.h"

#include <cmath>
#include <limits>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "kernels/spec.h"
#include "kernels/spec/warp.h"

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

TEST_CASE("Le niveau 1 en deux sources (image, caractéristiques) rend le splat de leur concaténation",
          "[kernels][gmfss]") {
  // Export de version 2 : la paire sort l'image (3 plans) et les caractéristiques séparées ; chaque plan
  // est normalisé par le poids de son seul pixel, donc deux splats du même mouvement dans les tranches de
  // warped{k}_1 valent le splat des 67 plans (ici 3 + 4).
  const std::vector<PlaneF> image{ramp(9, 5, 0.0F), ramp(9, 5, 0.25F), ramp(9, 5, 0.5F)};
  const std::vector<PlaneF> features{ramp(9, 5, 1.0F), ramp(9, 5, -3.0F), ramp(9, 5, 7.5F), ramp(9, 5, 2.0F)};
  std::vector<PlaneF> joined = image;
  joined.insert(joined.end(), features.begin(), features.end());
  ref::SplatMotion motion = uniform(9, 5, 1.5F, -0.75F, 0.5F);
  motion.planes[2].at(3, 2) = -4.0F;
  motion.planes[0].at(4, 1) = 2.25F;
  for (const float s : {0.3F, 1.0F}) {
    const auto whole = ref::soft_splat(joined, motion, s);
    const auto first = ref::soft_splat(image, motion, s);
    const auto second = ref::soft_splat(features, motion, s);
    REQUIRE(whole);
    REQUIRE(first);
    REQUIRE(second);
    REQUIRE(whole->size() == first->size() + second->size());
    for (std::size_t c = 0; c < whole->size(); ++c) {
      const PlaneF& part = c < first->size() ? (*first)[c] : (*second)[c - first->size()];
      CHECK((*whole)[c].data == part.data);  // même ordre des additions : au bit près
    }
  }
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

TEST_CASE("Le mouvement sur une grille deux fois plus grosse se lit entre ses pixels", "[kernels][gmfss]") {
  // Source 4 × 1 ; mouvement 2 × 1 : fx = 0 puis 2 pixels du mouvement, soit 0 puis 4 pixels source.
  // Le pixel 0 lit u = −1/4 (borné à 0) : il reste ; le pixel 1 lit u = 1/4 : 1/2 × 2 = 1 pixel, il
  // tombe en 2 ; le pixel 2 lit u = 3/4 : 3 pixels, hors du plan ; le pixel 3 lit u = 5/4 (borné à
  // 1) : 4 pixels, hors du plan.
  PlaneF value(4, 1, 0.0F);
  for (int x = 0; x < 4; ++x) value.at(x, 0) = static_cast<float>(x + 1);
  ref::SplatMotion motion = uniform(2, 1, 0.0F, 0.0F, 0.0F);
  motion.planes[0].at(1, 0) = 2.0F;
  const auto out = ref::soft_splat_at({value}, motion, 1.0F, MotionGrid{.scale = 2.0F});
  REQUIRE(out);
  CHECK_THAT(out->planes[0].at(0, 0), WithinAbs(1.0, 1e-5));
  CHECK_THAT(out->planes[0].at(1, 0), WithinAbs(0.0, 1e-6));
  CHECK_THAT(out->planes[0].at(2, 0), WithinAbs(2.0, 1e-5));
  CHECK_THAT(out->planes[0].at(3, 0), WithinAbs(0.0, 1e-6));
  // La couverture dit où une source arrive.
  CHECK_THAT(out->coverage.at(0, 0), WithinAbs(1.0, 1e-5));
  CHECK(out->coverage.at(1, 0) == 0.0F);
  CHECK_THAT(out->coverage.at(2, 0), WithinAbs(1.0, 1e-5));
  CHECK(out->coverage.at(3, 0) == 0.0F);
  // Un décalage d'un pixel du mouvement (spec : u = (x + 1/2) / scale − 1/2 + ox) : le pixel 0 lit
  // u = 3/4, soit 3/2 × 2 = 3 pixels, il tombe en 3 ; les pixels 1 à 3 lisent u ≥ 1 (borné à 1) :
  // 4 pixels, hors du plan.
  const auto shifted = ref::soft_splat_at({value}, motion, 1.0F, MotionGrid{.scale = 2.0F, .x = 1.0F});
  REQUIRE(shifted);
  for (int x = 0; x < 3; ++x) CHECK(shifted->planes[0].at(x, 0) == 0.0F);
  CHECK_THAT(shifted->planes[0].at(3, 0), WithinAbs(1.0, 1e-5));
  CHECK_THAT(shifted->coverage.at(3, 0), WithinAbs(1.0, 1e-5));
  // Sur la même grille, sans décalage : le splat ordinaire, couverture en plus.
  const auto same = ref::soft_splat_at({value}, uniform(4, 1, 1.0F, 0.0F, 0.0F), 0.5F, MotionGrid{});
  const auto plain = ref::soft_splat({value}, uniform(4, 1, 1.0F, 0.0F, 0.0F), 0.5F);
  REQUIRE((same && plain));
  CHECK(same->planes[0] == (*plain)[0]);
  CHECK(ref::soft_splat_at({value}, motion, 0.5F, MotionGrid{.scale = 0.0F}).error().code ==
        ErrorCode::ParameterOutOfRange);
}

TEST_CASE("La composition après SPAN rend la fusion où rien n'arrive, le détail agrandi ailleurs",
          "[kernels][gmfss]") {
  // Travail 2 × 1, sortie 4 × 2 ; la fusion 4 × 3 avec sa fenêtre 2 × 1 en (1, 1), du 9 autour.
  const auto splatted = [](int w, int h, float value, float coverage) {
    return ref::Splatted{.planes = {PlaneF(w, h, value), PlaneF(w, h, value), PlaneF(w, h, value)},
                         .coverage = PlaneF(w, h, coverage)};
  };
  std::vector<PlaneF> fusion(3, PlaneF(4, 3, 9.0F));
  const Window window{.x = 1, .y = 1, .width = 2, .height = 1};
  for (PlaneF& p : fusion) {
    p.at(1, 1) = 0.6F;
    p.at(2, 1) = 0.6F;
  }
  const float t = 0.25F;
  SECTION("les deux côtés arrivent : H + F − L") {
    const auto out =
        ref::gmfss_compose(splatted(4, 2, 0.5F, 1.0F), splatted(4, 2, 0.3F, 1.0F), splatted(2, 1, 0.4F, 1.0F),
                           splatted(2, 1, 0.2F, 1.0F), fusion, window, t);
    REQUIRE(out);
    // H = 0,75 × 0,5 + 0,25 × 0,3 = 0,45 ; L = 0,75 × 0,4 + 0,25 × 0,2 = 0,35 ; 0,45 + 0,6 − 0,35.
    for (const PlaneF& p : *out) {
      for (const float v : p.data) CHECK_THAT(v, WithinAbs(0.7, 1e-4));
    }
  }
  SECTION("rien n'arrive : la fusion agrandie seule") {
    const auto out =
        ref::gmfss_compose(splatted(4, 2, 0.0F, 0.0F), splatted(4, 2, 0.0F, 0.0F), splatted(2, 1, 0.0F, 0.0F),
                           splatted(2, 1, 0.0F, 0.0F), fusion, window, t);
    REQUIRE(out);
    for (const float v : (*out)[0].data) CHECK_THAT(v, WithinAbs(0.6, 1e-5));
  }
  SECTION("un seul côté arrive : son détail, quel que soit l'instant") {
    const auto out =
        ref::gmfss_compose(splatted(4, 2, 0.9F, 1.0F), splatted(4, 2, 0.0F, 0.0F), splatted(2, 1, 0.7F, 1.0F),
                           splatted(2, 1, 0.0F, 0.0F), fusion, window, t);
    REQUIRE(out);
    for (const float v : (*out)[1].data) CHECK_THAT(v, WithinAbs(0.8, 1e-4));  // 0,9 + 0,6 − 0,7
  }
  SECTION("la sortie reste dans [0, 1]") {
    const auto out =
        ref::gmfss_compose(splatted(4, 2, 1.0F, 1.0F), splatted(4, 2, 1.0F, 1.0F), splatted(2, 1, 0.0F, 1.0F),
                           splatted(2, 1, 0.0F, 1.0F), fusion, window, t);
    REQUIRE(out);
    for (const float v : (*out)[2].data) CHECK(v == 1.0F);
  }
  SECTION("refus") {
    const auto a = splatted(4, 2, 0.5F, 1.0F);
    const auto l = splatted(2, 1, 0.5F, 1.0F);
    CHECK(ref::gmfss_compose(a, a, l, l, fusion, window, 1.5F).error().code ==
          ErrorCode::ParameterOutOfRange);
    CHECK(ref::gmfss_compose(splatted(3, 2, 0.5F, 1.0F), a, l, l, fusion, window, t).error().code ==
          ErrorCode::SizeMismatch);
    CHECK(ref::gmfss_compose(a, a, l, l, fusion, Window{.x = 3, .y = 1, .width = 2, .height = 1}, t)
              .error()
              .code == ErrorCode::ParameterOutOfRange);
    CHECK(ref::gmfss_compose(a, a, l, l, {fusion[0], fusion[1]}, window, t).error().code ==
          ErrorCode::SizeMismatch);
  }
}
