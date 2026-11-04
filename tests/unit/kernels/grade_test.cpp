// La référence du grade exact (kernels/ref/grade.h) : neutre = identité, gris monotones, ancrage de la peau,
// cartes (tailles, max local, lissage de la zone) et refus.
#include "kernels/ref/grade.h"

#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "kernels/spec/grade_pixel.h"

using namespace ajp::kernels;
using Catch::Matchers::WithinAbs;

namespace {

// Un grade qui ne change rien : ni pied, ni gain, ni vibrance, ni fin de grade, ni noir, ni look.
GradeUniforms neutral() {
  GradeUniforms u;
  u.shadows = 0.0F;
  u.area_gain = 1.0F;
  u.spec_gain = 1.0F;
  u.vibrance = 0.0F;
  u.skin_on = 0.0F;
  return u;
}

RgbImage flat(int w, int h, Rgb c) {
  RgbImage image(w, h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      image.planes[0].at(x, y) = c.r;
      image.planes[1].at(x, y) = c.g;
      image.planes[2].at(x, y) = c.b;
    }
  return image;
}

// Un motif varié : rampe horizontale, couleurs par bandes, un éclat.
RgbImage pattern(int w, int h) {
  RgbImage image(w, h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const float t = static_cast<float>(x) / static_cast<float>(w - 1);
      const int band = (y * 4) / h;
      Rgb c{.r = t, .g = t, .b = t};
      if (band == 1) c = {.r = t, .g = 0.3F * t, .b = 0.1F};
      if (band == 2) c = {.r = 0.1F, .g = 0.4F * t, .b = t};
      image.planes[0].at(x, y) = c.r;
      image.planes[1].at(x, y) = c.g;
      image.planes[2].at(x, y) = c.b;
    }
  image.planes[0].at(w / 2, h - 2) = 1.0F;
  image.planes[1].at(w / 2, h - 2) = 1.0F;
  image.planes[2].at(w / 2, h - 2) = 1.0F;
  return image;
}

RgbImage graded(const RgbImage& pq, const GradeUniforms& u) {
  const auto maps = ref::grade_maps(pq, u, {}, {}, 1.0F / 24.0F);
  REQUIRE(maps);
  auto out = ref::grade_image(pq, *maps, u, {});
  REQUIRE(out);
  return *out;
}

RgbImage pq_of(const RgbImage& sdr) {
  auto pq = ref::apply_input(sdr, ref::sdr_input());
  REQUIRE(pq);
  return *pq;
}

}  // namespace

TEST_CASE("grade : la taille des cartes est celle de libplacebo (arrondi au plus proche)",
          "[kernels][grade]") {
  CHECK(grade::map_size(1920, 8) == 240);
  CHECK(grade::map_size(1080, 8) == 135);
  CHECK(grade::map_size(135, 4) == 34);    // 33,75
  CHECK(grade::map_size(1916, 8) == 240);  // 239,5 : la moitié vers le haut, comme roundf
  CHECK(grade::map_size(3, 8) == 1);
}

TEST_CASE("grade : neutre = identité à 1e-4 PQ", "[kernels][grade]") {
  const RgbImage pq = pq_of(pattern(64, 48));
  for (const float p3 : {0.0F, 1.0F}) {
    GradeUniforms u = neutral();
    u.gamut_p3 = p3;
    const RgbImage out = graded(pq, u);
    float worst = 0.0F;
    for (std::size_t c = 0; c < 3; ++c)
      for (std::size_t i = 0; i < pq.planes[c].data.size(); ++i)
        worst = std::max(worst, std::abs(out.planes[c].data[i] - pq.planes[c].data[i]));
    INFO("gamut_p3 = " << p3);
    CHECK(worst <= 1e-4F);
  }
}

TEST_CASE("grade : les gris restent monotones et gris", "[kernels][grade]") {
  GradeUniforms u;  // les défauts : pied, gains, vibrance
  float previous = -1.0F;
  for (int i = 0; i <= 64; ++i) {
    const float v = static_cast<float>(i) / 64.0F;
    const RgbImage out = graded(pq_of(flat(16, 16, {.r = v, .g = v, .b = v})), u);
    const Rgb c = out.at(8, 8);
    INFO("gris " << v);
    CHECK(c.r >= previous);
    CHECK_THAT(c.g, WithinAbs(c.r, 2e-4));
    CHECK_THAT(c.b, WithinAbs(c.r, 2e-4));
    previous = c.r;
  }
}

TEST_CASE("grade : la peau ne suit la scène qu'à 40 %", "[kernels][grade]") {
  // Une peau claire d'anime (sRGB 255, 224, 200) : son masque vaut 1.
  const RgbImage pq = pq_of(flat(32, 32, {.r = 1.0F, .g = 224.0F / 255.0F, .b = 200.0F / 255.0F}));
  GradeUniforms u;
  u.emulate = 1.0F;
  u.ad_on = 1.0F;
  u.sh_on = 1.0F;
  const auto fetch = [&pq](int x, int y) {
    const Rgb c = pq.at(std::clamp(x, 0, pq.width() - 1), std::clamp(y, 0, pq.height() - 1));
    return grade::V3{c.r, c.g, c.b};
  };
  REQUIRE(grade::skin_area(fetch, 16, 16, u) > 0.99F);
  const auto at = [&](float t, float skin_on) {
    GradeUniforms v = u;
    v.sh_t = t;
    v.skin_on = skin_on;
    return graded(pq, v).at(16, 16);
  };
  const auto lum = [](Rgb c) { return 0.2627F * c.r + 0.6780F * c.g + 0.0593F * c.b; };
  const float swing_skin = std::abs(lum(at(1.0F, 1.0F)) - lum(at(0.0F, 1.0F)));
  const float swing_decor = std::abs(lum(at(1.0F, 0.0F)) - lum(at(0.0F, 0.0F)));
  REQUIRE(swing_decor > 0.005F);
  CHECK(swing_skin < 0.7F * swing_decor);
}

TEST_CASE("grade : le max local prend le pixel le plus clair du bloc, gradé", "[kernels][grade]") {
  RgbImage sdr = flat(32, 16, {.r = 0.05F, .g = 0.05F, .b = 0.05F});
  sdr.planes[1].at(11, 5) = 0.9F;  // dans le bloc (1, 0)
  const RgbImage pq = pq_of(sdr);
  GradeUniforms u;
  u.black_lo = 0.05F;
  u.black_hi = 2.0F;
  const auto maps = ref::grade_maps(pq, u, {}, {}, 1.0F / 24.0F);
  REQUIRE(maps);
  REQUIRE(maps->local_w == 4);
  REQUIRE(maps->local_h == 2);
  const float dark = maps->localmax[0];
  const float lit = maps->localmax[1];
  CHECK(lit > dark + 0.1F);
  // Sans vrai noir : 0, sans rien lire.
  u.black_hi = 0.0F;
  const auto off = ref::grade_maps(pq, u, {}, {}, 1.0F / 24.0F);
  REQUIRE(off);
  for (const float v : off->localmax) CHECK(v == 0.0F);
}

TEST_CASE("grade : la zone se lisse à α = 1 − 0,7^(48 Δt) et repart d'une coupe", "[kernels][grade]") {
  const GradeUniforms u;
  const RgbImage dim = pq_of(flat(256, 128, {.r = 0.30F, .g = 0.30F, .b = 0.30F}));
  const RgbImage near = pq_of(flat(256, 128, {.r = 0.34F, .g = 0.34F, .b = 0.34F}));
  const RgbImage cut = pq_of(flat(256, 128, {.r = 0.95F, .g = 0.95F, .b = 0.95F}));
  const auto first = ref::grade_maps(dim, u, {}, {}, 1.0F / 24.0F);
  REQUIRE(first);
  const auto measured = ref::grade_maps(near, u, {}, {}, 1.0F / 24.0F);
  REQUIRE(measured);
  const float dt = 1.0F / 24.0F;
  const auto smoothed = ref::grade_maps(near, u, {}, first->history, dt);
  REQUIRE(smoothed);
  const float alpha = 1.0F - std::pow(0.7F, 48.0F * dt);
  const float a = first->area[0];
  const float b = measured->area[0];
  REQUIRE(std::abs(b - a) <= grade::kSmoothJump);
  CHECK_THAT(smoothed->area[0], WithinAbs(a * (1.0F - alpha) + b * alpha, 1e-6));
  // Une coupe (> 0,12 en PQ) : la mesure, sans traînée.
  const auto jumped = ref::grade_maps(cut, u, {}, first->history, dt);
  const auto cut_alone = ref::grade_maps(cut, u, {}, {}, dt);
  REQUIRE(jumped);
  REQUIRE(cut_alone);
  CHECK(jumped->area == cut_alone->area);
}

TEST_CASE("grade : le look, le vrai noir et KWin s'appliquent dans cet ordre", "[kernels][grade]") {
  // Un look qui inverse les couleurs, à mi-intensité.
  const int n = 2;
  std::vector<float> lut;
  for (int b = 0; b < n; ++b)
    for (int gr = 0; gr < n; ++gr)
      for (int r = 0; r < n; ++r) {
        lut.push_back(1.0F - static_cast<float>(r));
        lut.push_back(1.0F - static_cast<float>(gr));
        lut.push_back(1.0F - static_cast<float>(b));
      }
  const ref::GradeLook look{.size = n, .rgb = lut};
  GradeUniforms u = neutral();
  u.look_mix = 0.5F;
  const Rgb code{.r = 0.5F, .g = 0.4F, .b = 0.3F};
  const Rgb looked = ref::grade_pixel(code, 0.0F, 0.4F, 0.4F, 0.0F, u, look);
  const auto lut_fetch = [&lut](grade::V3 c) {
    // n = 2 : la lecture trilinéaire d'une inversion est l'inversion.
    (void)lut;
    return grade::V3{1.0F - c.r, 1.0F - c.g, 1.0F - c.b};
  };
  const grade::V3 expect =
      grade::look_code(grade::grade_code({code.r, code.g, code.b}, 0.0F, 0.4F, 0.4F, u), lut_fetch, u);
  CHECK_THAT(looked.r, WithinAbs(expect.r, 1e-5));
  CHECK_THAT(looked.g, WithinAbs(expect.g, 1e-5));
  CHECK_THAT(looked.b, WithinAbs(expect.b, 1e-5));
  // Vrai noir : un pixel sombre dans un bloc sombre tombe à 0.
  GradeUniforms black = neutral();
  black.black_lo = 1.0F;
  black.black_hi = 3.0F;
  const Rgb dark{.r = 0.05F, .g = 0.05F, .b = 0.05F};  // ~0,1 nit
  const Rgb crushed = ref::grade_pixel(dark, 0.0F, 0.0F, 0.0F, 0.05F, black, {});
  CHECK(crushed.r < 1e-6F);
  // Un pixel sombre à côté d'un éclat (max local haut) n'est pas touché.
  const Rgb kept = ref::grade_pixel(dark, 0.0F, 0.0F, 0.0F, 0.6F, black, {});
  CHECK_THAT(kept.r, WithinAbs(0.05, 1e-4));
  // KWin : le blanc PQ de 203 nits reste sous le pic, une couleur de 10 000 nits est ramenée au pic.
  GradeUniforms kwin = neutral();
  kwin.kwin_on = 1.0F;
  const Rgb top = ref::grade_pixel({.r = 1.0F, .g = 1.0F, .b = 1.0F}, 0.0F, 0.0F, 0.0F, 0.0F, kwin, {});
  const float top_nits = grade::pq_to_nits({top.r, top.g, top.b}).r * kwin.kwin_ref / 203.0F;
  CHECK(top_nits <= kwin.kwin_peak * 1.001F);
}

TEST_CASE("grade : les entrées hors contrat sont refusées", "[kernels][grade]") {
  const RgbImage pq = pq_of(flat(16, 16, {.r = 0.5F, .g = 0.5F, .b = 0.5F}));
  const GradeUniforms u;
  const std::vector<float> wrong(3, 0.0F);
  CHECK_FALSE(ref::grade_maps(pq, u, {}, wrong, 0.04F));
  const std::vector<float> lut(5, 0.0F);
  CHECK_FALSE(ref::grade_maps(pq, u, {.size = 2, .rgb = lut}, {}, 0.04F));
  const auto maps = ref::grade_maps(pq, u, {}, {}, 0.04F);
  REQUIRE(maps);
  CHECK_FALSE(ref::grade_image(pq_of(flat(32, 16, {.r = 0.5F, .g = 0.5F, .b = 0.5F})), *maps, u, {}));
  CHECK_FALSE(ref::grade_maps(RgbImage(1, 1), u, {}, {}, 0.04F));
}
