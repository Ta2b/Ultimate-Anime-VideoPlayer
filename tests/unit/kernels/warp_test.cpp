#include "kernels/ref/warp.h"

#include <cmath>
#include <limits>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "kernels/spec.h"

using namespace ajp::kernels;
using Catch::Matchers::WithinAbs;

namespace {

// Image dont chaque plan vaut x + offset : un déplacement horizontal se lit directement.
RgbImage ramp(int w, int h, float offset) {
  RgbImage image(w, h);
  for (PlaneF& p : image.planes) {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) p.at(x, y) = static_cast<float>(x) + offset;
    }
  }
  return image;
}

// Mouvement uniforme sur des plans pw × ph, fenêtre area ; hors fenêtre, des valeurs à ignorer.
ref::Motion uniform_motion(int pw, int ph, Window area, float f0x, float f1x, float mask) {
  ref::Motion m{
      .flow = {PlaneF(pw, ph, 99.0F), PlaneF(pw, ph, 99.0F), PlaneF(pw, ph, 99.0F), PlaneF(pw, ph, 99.0F)},
      .mask = PlaneF(pw, ph, 0.5F),
      .area = area};
  for (int y = area.y; y < area.y + area.height; ++y) {
    for (int x = area.x; x < area.x + area.width; ++x) {
      m.flow[0].at(x, y) = f0x;
      m.flow[1].at(x, y) = 0.0F;
      m.flow[2].at(x, y) = f1x;
      m.flow[3].at(x, y) = 0.0F;
      m.mask.at(x, y) = mask;
    }
  }
  return m;
}

}  // namespace

TEST_CASE("warp : échantillon bilinéaire borné au plan", "[warp]") {
  PlaneF p(4, 2);
  for (int y = 0; y < 2; ++y) {
    for (int x = 0; x < 4; ++x) p.at(x, y) = static_cast<float>(10 * y + x);
  }
  CHECK(ref::sample(p, 2.0F, 1.0F) == 12.0F);
  CHECK(ref::sample(p, 1.5F, 0.0F) == 1.5F);
  CHECK(ref::sample(p, 1.5F, 0.5F) == 6.5F);
  CHECK(ref::sample(p, -3.0F, 0.0F) == 0.0F);  // clamp à gauche
  CHECK(ref::sample(p, 9.0F, 5.0F) == 13.0F);  // clamp en bas à droite
  CHECK(ref::sample(p, std::numeric_limits<float>::quiet_NaN(), 1.0F) == 10.0F);
  // Dans une fenêtre : coordonnées relatives à son coin, clamp à ses bords.
  CHECK(ref::sample(p, Window{.x = 1, .y = 0, .width = 2, .height = 2}, 0.0F, 0.0F) == 1.0F);
  CHECK(ref::sample(p, Window{.x = 1, .y = 0, .width = 2, .height = 2}, 5.0F, 1.0F) == 12.0F);
}

TEST_CASE("warp : un échantillon hors contrat est refusé, jamais lu hors du plan", "[warp]") {
  const PlaneF p(4, 2, 1.0F);
  CHECK(ref::sample(PlaneF(), 0.0F, 0.0F).error().code == ErrorCode::DimensionOutOfRange);
  CHECK(ref::sample(p, Window{.x = 3, .y = 0, .width = 2, .height = 2}, 0.0F, 0.0F).error().code ==
        ErrorCode::ParameterOutOfRange);
  CHECK(ref::sample(p, Window{.x = 0, .y = 0, .width = 0, .height = 2}, 0.0F, 0.0F).error().code ==
        ErrorCode::ParameterOutOfRange);
  // Une largeur géante ne déborde pas l'addition du test de la fenêtre.
  CHECK(ref::sample(p, Window{.x = 2, .y = 0, .width = std::numeric_limits<int>::max(), .height = 1}, 0.0F,
                    0.0F)
            .error()
            .code == ErrorCode::ParameterOutOfRange);
  // Un plan dont les données ne couvrent pas les dimensions annoncées.
  PlaneF hollow = p;
  hollow.data.pop_back();
  CHECK(ref::sample(hollow, 0.0F, 0.0F).error().code == ErrorCode::SizeMismatch);
  ref::Motion motion = uniform_motion(8, 4, {0, 0, 8, 4}, 0.0F, 0.0F, 0.5F);
  motion.flow[1].data.clear();
  CHECK(ref::warp_blend(RgbImage(8, 4), RgbImage(8, 4), motion).error().code == ErrorCode::SizeMismatch);
}

TEST_CASE("warp : le masque choisit entre a et b", "[warp]") {
  const RgbImage a(8, 4, 0.2F);
  const RgbImage b(8, 4, 0.6F);
  const float mask = GENERATE(0.0F, 0.25F, 1.0F);
  const auto out = ref::warp_blend(a, b, uniform_motion(8, 4, {0, 0, 8, 4}, 0.0F, 0.0F, mask));
  REQUIRE(out);
  CHECK_THAT(out->planes[1].at(3, 2), WithinAbs(0.2 * mask + 0.6 * (1.0 - mask), 1e-7));
}

TEST_CASE("warp : déplacement arrière, bords en clamp", "[warp]") {
  // f0x = +2 : le pixel x lit a en x + 2 ; au bord droit, a est relu en 15.
  const auto out = ref::warp_blend(ramp(16, 4, 0.0F), ramp(16, 4, 0.0F),
                                   uniform_motion(16, 4, {0, 0, 16, 4}, 2.0F, 0.0F, 1.0F));
  REQUIRE(out);
  CHECK(out->planes[0].at(3, 1) == 5.0F);
  CHECK(out->planes[0].at(14, 1) == 15.0F);
  // Demi-pixel : lecture bilinéaire.
  const auto half = ref::warp_blend(ramp(16, 4, 0.0F), ramp(16, 4, 0.0F),
                                    uniform_motion(16, 4, {0, 0, 16, 4}, 0.5F, 0.0F, 1.0F));
  REQUIRE(half);
  CHECK(half->planes[2].at(6, 0) == 6.5F);
}

TEST_CASE("warp : mouvement estimé sur une fenêtre plus petite, mis à l'échelle", "[warp]") {
  // Image 16 × 8, mouvement 8 × 4 dans des plans 12 × 8 remplis : 1 px de fenêtre = 2 px d'image.
  const ref::Motion motion =
      uniform_motion(12, 8, {.x = 2, .y = 2, .width = 8, .height = 4}, 1.0F, 0.0F, 1.0F);
  const auto out = ref::warp_blend(ramp(16, 8, 0.0F), ramp(16, 8, 0.0F), motion);
  REQUIRE(out);
  CHECK(out->planes[0].at(4, 3) == 6.0F);
  const auto peak = ref::flow_max(motion);
  REQUIRE(peak);
  CHECK(*peak == 1.0F);  // les 99 hors de la fenêtre ne comptent pas
}

TEST_CASE("warp : à t = 1/2, warp_blend_at égale warp_blend", "[warp]") {
  const ref::Motion motion = uniform_motion(16, 4, {0, 0, 16, 4}, -1.5F, 1.5F, 0.3F);
  const auto mid = ref::warp_blend(ramp(16, 4, 0.0F), ramp(16, 4, -3.0F), motion);
  const auto at = ref::warp_blend_at(ramp(16, 4, 0.0F), ramp(16, 4, -3.0F), motion, 0.5F);
  REQUIRE(mid);
  REQUIRE(at);
  CHECK(*mid == *at);
}

TEST_CASE("warp : un instant quelconque suit la ligne droite du mouvement", "[warp]") {
  // Le contenu avance de D = 4 px de a vers b : b(x) = a(x − 4). Au milieu, RIFE estime
  // f0 = −2 et f1 = +2. À l'instant t, l'image attendue est a(x − tD), quel que soit le masque.
  const float t = GENERATE(0.25F, 0.75F);
  const ref::Motion motion = uniform_motion(32, 4, {0, 0, 32, 4}, -2.0F, 2.0F, 0.5F);
  const auto out = ref::warp_blend_at(ramp(32, 4, 0.0F), ramp(32, 4, -4.0F), motion, t);
  REQUIRE(out);
  for (int x = 6; x < 26; ++x) CHECK_THAT(out->planes[0].at(x, 2), WithinAbs(x - 4.0 * t, 1e-5));
}

TEST_CASE("warp : la loi de mélange ajustée sur RIFE", "[warp][parity]") {
  // Sans mouvement : sortie = a m + b (1 − m), logit(m) = A logit(mm) + B ln((1 − t)/t) + C.
  const float t = 0.25F;
  const float mm = 0.4F;
  const ref::Motion motion = uniform_motion(8, 4, {0, 0, 8, 4}, 0.0F, 0.0F, mm);
  const auto out = ref::warp_blend_at(RgbImage(8, 4, 1.0F), RgbImage(8, 4, 0.0F), motion, t);
  REQUIRE(out);
  const double z = 0.285 * std::log(0.4 / 0.6) + 2.591 * std::log(3.0) + 0.170;
  CHECK_THAT(out->planes[0].at(4, 2), WithinAbs(1.0 / (1.0 + std::exp(-z)), 1e-6));
  // Un masque saturé est borné à ε avant le logit.
  const auto saturated = ref::warp_blend_at(RgbImage(8, 4, 1.0F), RgbImage(8, 4, 0.0F),
                                            uniform_motion(8, 4, {0, 0, 8, 4}, 0.0F, 0.0F, 1.0F), t);
  REQUIRE(saturated);
  CHECK(std::isfinite(saturated->planes[0].at(0, 0)));
  CHECK(saturated->planes[0].at(0, 0) < 1.0F);
}

TEST_CASE("warp : seuil d'une seule passe RIFE par paire", "[warp]") {
  static_assert(single_run_motion(720) == 15.0F);
  CHECK(single_run_motion(1080) == 22.5F);
  CHECK(single_run_motion(540) == 11.25F);
}

TEST_CASE("warp : les entrées hors contrat sont refusées", "[warp]") {
  const ref::Motion motion = uniform_motion(8, 4, {0, 0, 8, 4}, 0.0F, 0.0F, 0.5F);
  const RgbImage a(8, 4);
  for (const float t : {0.0F, 1.0F, -0.5F, std::numeric_limits<float>::quiet_NaN()}) {
    CHECK(ref::warp_blend_at(a, a, motion, t).error().code == ErrorCode::ParameterOutOfRange);
  }
  CHECK(ref::warp_blend(a, RgbImage(8, 6), motion).error().code == ErrorCode::SizeMismatch);
  ref::Motion outside = motion;
  outside.area = {.x = 2, .y = 0, .width = 8, .height = 4};
  CHECK(ref::warp_blend(a, a, outside).error().code == ErrorCode::ParameterOutOfRange);
  ref::Motion uneven = motion;
  uneven.flow[3] = PlaneF(8, 5);
  CHECK(ref::flow_max(uneven).error().code == ErrorCode::SizeMismatch);
}

// Garder les poses (spec/poses.h) : b est a décalée de 6 pixels vers la droite, mouvement de la
// paire uniforme (f0 = −3, f1 = +3).
TEST_CASE("poses : un mouvement régulier garde l'instant de warp_blend_at", "[warp][poses]") {
  const RgbImage a = ramp(32, 4, 0.0F);
  const RgbImage b = ramp(32, 4, -6.0F);
  const ref::Motion motion = uniform_motion(32, 4, {0, 0, 32, 4}, -3.0F, 3.0F, 0.5F);
  for (const float t : {0.25F, 0.75F}) {
    const auto plain = ref::warp_blend_at(a, b, motion, t);
    const auto poses = ref::warp_blend_poses(a, b, motion, &motion, t);
    REQUIRE(plain);
    REQUIRE(poses);
    for (int x = 0; x < 32; ++x)
      CHECK_THAT(poses->planes[0].at(x, 2), WithinAbs(plain->planes[0].at(x, 2), 1e-5));
  }
}

TEST_CASE("poses : après une paire tenue, la pose tient jusqu'au milieu puis passe à b", "[warp][poses]") {
  const RgbImage a = ramp(32, 4, 0.0F);
  const RgbImage b = ramp(32, 4, -6.0F);
  const ref::Motion motion = uniform_motion(32, 4, {0, 0, 32, 4}, -3.0F, 3.0F, 0.5F);
  const auto early = ref::warp_blend_poses(a, b, motion, nullptr, 0.25F);
  const auto late = ref::warp_blend_poses(a, b, motion, nullptr, 0.75F);
  REQUIRE(early);
  REQUIRE(late);
  for (int x = 8; x < 24; ++x) {
    CHECK_THAT(early->planes[0].at(x, 2), WithinAbs(a.planes[0].at(x, 2), 0.05));
    CHECK_THAT(late->planes[0].at(x, 2), WithinAbs(b.planes[0].at(x, 2), 0.05));
  }
}

TEST_CASE("poses : un personnage en deux sur un fond en uns", "[warp][poses]") {
  // Paire précédente : la moitié gauche (le fond) bougeait déjà de 6, la droite (le personnage) non.
  const RgbImage a = ramp(64, 4, 0.0F);
  const RgbImage b = ramp(64, 4, -6.0F);
  const ref::Motion motion = uniform_motion(64, 4, {0, 0, 64, 4}, -3.0F, 3.0F, 0.5F);
  ref::Motion previous = uniform_motion(64, 4, {0, 0, 64, 4}, -3.0F, 3.0F, 0.5F);
  for (int y = 0; y < 4; ++y) {
    for (int x = 32; x < 64; ++x) {
      previous.flow[0].at(x, y) = 0.0F;
      previous.flow[2].at(x, y) = 0.0F;
    }
  }
  const auto plain = ref::warp_blend_at(a, b, motion, 0.25F);
  const auto poses = ref::warp_blend_poses(a, b, motion, &previous, 0.25F);
  REQUIRE(plain);
  REQUIRE(poses);
  for (int x = 8; x < 24; ++x)
    CHECK_THAT(poses->planes[0].at(x, 2), WithinAbs(plain->planes[0].at(x, 2), 1e-4));
  for (int x = 42; x < 58; ++x) CHECK_THAT(poses->planes[0].at(x, 2), WithinAbs(a.planes[0].at(x, 2), 0.05));
}

TEST_CASE("poses : instant et mouvements hors contrat refusés", "[warp][poses]") {
  const RgbImage a = ramp(8, 4, 0.0F);
  const ref::Motion motion = uniform_motion(8, 4, {0, 0, 8, 4}, 0.0F, 0.0F, 0.5F);
  CHECK(ref::warp_blend_poses(a, a, motion, nullptr, 1.0F).error().code == ErrorCode::ParameterOutOfRange);
  const ref::Motion other = uniform_motion(8, 4, {0, 0, 4, 4}, 0.0F, 0.0F, 0.5F);
  CHECK(ref::warp_blend_poses(a, a, motion, &other, 0.5F).error().code == ErrorCode::SizeMismatch);
  const ref::Motion smaller = uniform_motion(4, 4, {0, 0, 4, 4}, 0.0F, 0.0F, 0.5F);
  CHECK(ref::warp_blend_poses(a, a, motion, &smaller, 0.5F).error().code == ErrorCode::SizeMismatch);
  CHECK(pose_time(0.25F, 0.5F) == 0.25F);
  CHECK(pose_time(0.75F, 0.5F) == 0.75F);
  CHECK(pose_time(0.5F, 0.0F) == 1.0F);
}
