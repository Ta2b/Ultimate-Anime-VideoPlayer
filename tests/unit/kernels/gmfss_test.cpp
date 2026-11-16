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

TEST_CASE("Splat sur sa grille : à la même taille, le splat tel quel, couverture comprise",
          "[kernels][gmfss]") {
  const std::vector<PlaneF> source{ramp(6, 4, 0.0F)};
  const ref::SplatMotion motion = uniform(6, 4, 2.0F, 0.0F, 1.0F);
  auto plain = ref::soft_splat(source, motion, 0.5F);
  auto grid = ref::soft_splat_at(source, motion, 0.5F, MotionGrid{});
  REQUIRE(plain);
  REQUIRE(grid);
  for (std::size_t i = 0; i < source[0].data.size(); ++i)
    CHECK_THAT(grid->planes[0].data[i], WithinAbs((*plain)[0].data[i], 1e-6));
  // Flot d'un pixel à s = 1/2 : la première colonne ne reçoit rien.
  CHECK_THAT(grid->coverage.at(0, 1), WithinAbs(0.0, 1e-6));
  CHECK_THAT(grid->coverage.at(3, 1), WithinAbs(1.0, 1e-5));
}

TEST_CASE("Splat sur sa grille : le flot d'une grille deux fois plus petite, mis à l'échelle",
          "[kernels][gmfss]") {
  // Mouvement de 2 pixels sur la grille 4 × 2 : 4 pixels sur la source 8 × 4, à s = 1/2 deux pixels.
  const std::vector<PlaneF> source{ramp(8, 4, 1.0F)};
  auto out = ref::soft_splat_at(source, uniform(4, 2, 2.0F, 0.0F, 0.0F), 0.5F, MotionGrid{.scale = 2.0F});
  REQUIRE(out);
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 8; ++x) {
      const float expected = x >= 2 ? source[0].at(x - 2, y) : 0.0F;
      CHECK_THAT(out->planes[0].at(x, y), WithinAbs(expected, 1e-4));
    }
  }
}

TEST_CASE("Composition : le détail du travail, corrigé du résidu de la fusion", "[kernels][gmfss]") {
  // Sans mouvement : H est la source de travail, L la source de RIFE ; out = H + up(F − L).
  const int w = 4;
  const int h = 2;
  const auto still = [](int sw, int sh, float value) {
    const std::vector<PlaneF> planes(3, PlaneF(sw, sh, value));
    return *ref::soft_splat_at(planes, uniform(sw, sh, 0.0F, 0.0F, 0.0F), 0.5F, MotionGrid{});
  };
  const ref::Splatted high = still(6, 3, 0.4F);
  const ref::Splatted low = still(w, h, 0.3F);
  std::vector<PlaneF> fusion(3, PlaneF(w + 2, h + 2, 0.0F));
  for (PlaneF& p : fusion) {
    for (int y = 1; y <= h; ++y) {
      for (int x = 1; x <= w; ++x) p.at(x, y) = 0.35F;
    }
  }
  auto out =
      ref::gmfss_compose(high, high, low, low, fusion, Window{.x = 1, .y = 1, .width = w, .height = h}, 0.5F);
  REQUIRE(out);
  REQUIRE((*out)[0].width == 6);
  for (const PlaneF& p : *out) {
    for (const float v : p.data) CHECK_THAT(v, WithinAbs(0.45, 1e-5));
  }
  // Une sortie plus petite que RIFE, ou une fenêtre d'une autre taille : refus.
  CHECK_FALSE(ref::gmfss_compose(low, low, high, high, fusion,
                                 Window{.x = 1, .y = 1, .width = 6, .height = 3}, 0.5F));
  CHECK_FALSE(ref::gmfss_compose(high, high, low, low, fusion,
                                 Window{.x = 0, .y = 0, .width = 3, .height = 2}, 0.5F));
}

TEST_CASE("Décor fixe : la clé reprend les sources là où rien ne bouge", "[kernels][gmfss]") {
  // a et b égaux à gauche, très différents sur la colonne de droite ; la clé, plus douce, vaut 0,2.
  const int w = 6;
  const int h = 3;
  std::vector<PlaneF> a(3, PlaneF(w, h, 0.5F));
  std::vector<PlaneF> b = a;
  for (PlaneF& p : b) {
    for (int y = 0; y < h; ++y) p.at(w - 1, y) = 0.9F;
  }
  const std::vector<PlaneF> key(3, PlaneF(w, h, 0.2F));
  auto out = ref::gmfss_still(key, a, b, 0.25F);
  REQUIRE(out);
  for (const PlaneF& p : *out) {
    for (int y = 0; y < h; ++y) {
      // Loin de l'écart : les sources exactement ; à côté ou dessus : la clé telle quelle.
      for (int x = 0; x < w - 2; ++x) CHECK_THAT(p.at(x, y), WithinAbs(0.5, 1e-6));
      CHECK_THAT(p.at(w - 2, y), WithinAbs(0.2, 1e-6));
      CHECK_THAT(p.at(w - 1, y), WithinAbs(0.2, 1e-6));
    }
  }
  // Un écart entre les deux seuils : à mi-chemin entre la clé et le mélange des sources.
  std::vector<PlaneF> near = a;
  const float gap = (kGmfssStillLow + kGmfssStillHigh) / 2.0F;
  for (PlaneF& p : near) p = PlaneF(w, h, 0.5F + gap);
  auto half = ref::gmfss_still(key, a, near, 0.0F);
  REQUIRE(half);
  CHECK_THAT((*half)[0].at(2, 1), WithinAbs(0.2F + 0.5F * (0.5F - 0.2F), 1e-5));
  // Tailles différentes ou instant hors de [0, 1] : refus.
  CHECK_FALSE(ref::gmfss_still(key, std::vector<PlaneF>(3, PlaneF(w - 1, h, 0.5F)), b, 0.5F));
  CHECK_FALSE(ref::gmfss_still(key, a, b, 1.5F));
}
