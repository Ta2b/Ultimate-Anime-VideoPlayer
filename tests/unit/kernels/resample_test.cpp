#include "kernels/ref/resample.h"

#include <cmath>
#include <span>
#include <tuple>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "kernels/spec.h"

using namespace ajp::kernels;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

void check_row(std::span<const float> row, const std::vector<double>& want) {
  REQUIRE(row.size() == want.size());
  for (std::size_t j = 0; j < want.size(); ++j) CHECK_THAT(row[j], WithinAbs(want[j], 1e-7));
}

}  // namespace

TEST_CASE("rééchantillonnage : le noyau Spline36 et ses fractions", "[resample]") {
  static_assert(spline36(0.0) == 1.0);
  static_assert(spline36(3.0) == 0.0);
  static_assert(bilinear(0.25) == 0.75);
  CHECK_THAT(spline36(1.0), WithinAbs(0.0, 1e-15));
  CHECK_THAT(spline36(2.0), WithinAbs(0.0, 1e-15));
  // Aux demi-entiers : 91/152, −18/152, 3/152, de somme 1/2 de chaque côté.
  CHECK_THAT(spline36(0.5), WithinAbs(91.0 / 152.0, 1e-15));
  CHECK_THAT(spline36(-1.5), WithinAbs(-18.0 / 152.0, 1e-15));
  CHECK_THAT(spline36(2.5), WithinAbs(3.0 / 152.0, 1e-15));
  CHECK(spline36(1.3) == spline36(-1.3));
  CHECK(support(Filter::Spline36) == 3.0);
  CHECK(support(Filter::Bilinear) == 1.0);
}

TEST_CASE("rééchantillonnage : bords en miroir puis clamp", "[resample]") {
  static_assert(mirror_index(-1, 4) == 0);
  static_assert(mirror_index(-2, 4) == 1);
  static_assert(mirror_index(4, 4) == 3);
  static_assert(mirror_index(5, 4) == 2);
  static_assert(mirror_index(2, 4) == 2);
  static_assert(mirror_index(-3, 1) == 0);
  static_assert(mirror_index(5, 1) == 0);
  static_assert(mirror_index(-9, 2) == 0);
}

TEST_CASE("rééchantillonnage : poids connus, calculés à la main", "[resample][parity]") {
  SECTION("Spline36 ×2, chroma en siting gauche") {
    const auto w = ref::compute_weights(2, 4, chroma_shift_up(Siting::Left).x, Filter::Spline36);
    REQUIRE(w);
    CHECK(w->taps == 6);
    CHECK(w->start == std::vector<int>{-2, -2, -1, -1});
    check_row(w->row(0), {0, 0, 1, 0, 0, 0});
    check_row(w->row(1), {3 / 152.0, -18 / 152.0, 91 / 152.0, 91 / 152.0, -18 / 152.0, 3 / 152.0});
    check_row(w->row(2), {0, 0, 1, 0, 0, 0});
  }
  SECTION("bilinéaire ×2 : (1/4, 3/4) puis (3/4, 1/4)") {
    const auto w = ref::compute_weights(4, 8, 0.0, Filter::Bilinear);
    REQUIRE(w);
    CHECK(w->taps == 2);
    CHECK(w->start[0] == -1);
    check_row(w->row(0), {0.25, 0.75});
    check_row(w->row(1), {0.75, 0.25});
  }
  SECTION("Area 3:2 : (2/3, 1/3) puis (1/3, 2/3), comme le noyau 1080 vers 720") {
    const auto w = ref::compute_weights(6, 4, 0.0, Filter::Area);
    REQUIRE(w);
    CHECK(w->taps == 2);
    CHECK(w->start == std::vector<int>{0, 1, 3, 4});
    check_row(w->row(0), {2 / 3.0, 1 / 3.0});
    check_row(w->row(1), {1 / 3.0, 2 / 3.0});
  }
  SECTION("Spline36 1080 vers 720 : noyau étiré, 10 prises") {
    const auto w = ref::compute_weights(1080, 720, 0.0, Filter::Spline36);
    REQUIRE(w);
    CHECK(w->taps == 10);
  }
}

TEST_CASE("rééchantillonnage : chaque table somme à 1", "[resample]") {
  const auto [src, dst, shift, filter] = GENERATE(table<int, int, double, Filter>({
      {960, 1920, 0.25, Filter::Spline36},
      {540, 1080, 0.0, Filter::Spline36},
      {1080, 720, 0.0, Filter::Spline36},
      {1920, 3840, 0.0, Filter::Spline36},
      {1920, 960, -0.5, Filter::Spline36},
      {683, 1366, 0.25, Filter::Bilinear},
      {1280, 640, -0.5, Filter::Bilinear},
      {1080, 720, 0.0, Filter::Area},
      {3840, 1280, 0.0, Filter::Area},
      {1366, 854, 0.0, Filter::Area},
  }));
  const auto w = ref::compute_weights(src, dst, shift, filter);
  REQUIRE(w);
  for (int i = 0; i < dst; ++i) {
    double sum = 0.0;
    for (const float x : w->row(i)) sum += x;
    REQUIRE_THAT(sum, WithinAbs(1.0, tolerance::kWeightSum));
  }
}

TEST_CASE("rééchantillonnage : une image uniforme est conservée", "[resample]") {
  const auto [sw, sh, dw, dh, filter] = GENERATE(table<int, int, int, int, Filter>({
      {64, 36, 128, 72, Filter::Spline36},
      {96, 54, 64, 36, Filter::Spline36},
      {96, 54, 64, 36, Filter::Area},
      {128, 72, 32, 18, Filter::Area},
      {40, 30, 80, 60, Filter::Bilinear},
      {80, 60, 30, 22, Filter::Bilinear},
  }));
  const float value = 0.7071F;
  const auto out = ref::resize(PlaneF(sw, sh, value), dw, dh, filter);
  REQUIRE(out);
  CHECK(out->width == dw);
  CHECK(out->height == dh);
  for (const float v : out->data) REQUIRE_THAT(v, WithinRel(value, tolerance::kConstantImage));
}

TEST_CASE("rééchantillonnage : Area 2:1 donne la moyenne de chaque bloc 2 × 2", "[resample][parity]") {
  PlaneF in(4, 4);
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) in.at(x, y) = static_cast<float>(x + 4 * y);
  }
  const auto out = ref::resize(in, 2, 2, Filter::Area);
  REQUIRE(out);
  CHECK(out->at(0, 0) == 2.5F);   // (0 + 1 + 4 + 5) / 4
  CHECK(out->at(1, 0) == 4.5F);   // (2 + 3 + 6 + 7) / 4
  CHECK(out->at(0, 1) == 10.5F);  // (8 + 9 + 12 + 13) / 4
  CHECK(out->at(1, 1) == 12.5F);
}

TEST_CASE("rééchantillonnage : RGB passe plan par plan, horizontal puis vertical", "[resample]") {
  RgbImage in(8, 6, 0.25F);
  in.planes[1] = PlaneF(8, 6, 0.5F);
  const auto out = ref::resize(in, 4, 4, Filter::Spline36);
  REQUIRE(out);
  CHECK(out->width() == 4);
  CHECK_THAT(out->planes[0].at(2, 1), WithinRel(0.25, 1e-6));
  CHECK_THAT(out->planes[1].at(2, 1), WithinRel(0.5, 1e-6));
  // Passes séparées : même résultat que resize.
  const auto wx = ref::compute_weights(8, 4, 0.0, Filter::Spline36);
  const auto wy = ref::compute_weights(6, 4, 0.0, Filter::Spline36);
  REQUIRE(wx);
  REQUIRE(wy);
  const auto rows = ref::resample_rows(in.planes[2], *wx);
  REQUIRE(rows);
  const auto both = ref::resample_columns(*rows, *wy);
  REQUIRE(both);
  CHECK(*both == out->planes[2]);
}

TEST_CASE("rééchantillonnage : les entrées hors contrat sont refusées", "[resample]") {
  CHECK(ref::compute_weights(720, 1080, 0.0, Filter::Area).error().code == ErrorCode::FilterDirection);
  CHECK(ref::compute_weights(1080, 720, 0.25, Filter::Area).error().code == ErrorCode::ParameterOutOfRange);
  CHECK(ref::compute_weights(0, 720, 0.0, Filter::Spline36).error().code == ErrorCode::DimensionOutOfRange);
  CHECK(ref::compute_weights(720, 20000, 0.0, Filter::Spline36).error().code ==
        ErrorCode::DimensionOutOfRange);
  CHECK(ref::compute_weights(720, 360, 1.5, Filter::Spline36).error().code == ErrorCode::ParameterOutOfRange);
  CHECK(ref::compute_weights(720, 360, std::nan(""), Filter::Bilinear).error().code ==
        ErrorCode::ParameterOutOfRange);
  CHECK(ref::resize(PlaneF(8, 8), 5, 4, Filter::Spline36).error().code == ErrorCode::OddDimension);
  CHECK(ref::resize(PlaneF(7, 8), 4, 4, Filter::Spline36).error().code == ErrorCode::OddDimension);
  const auto w = ref::compute_weights(10, 5, 0.0, Filter::Spline36);
  REQUIRE(w);
  CHECK(ref::resample_rows(PlaneF(8, 2), *w).error().code == ErrorCode::SizeMismatch);
  CHECK(ref::resample_columns(PlaneF(2, 8), *w).error().code == ErrorCode::SizeMismatch);
  CHECK(name(ErrorCode::FilterDirection) == "FilterDirection");
}

TEST_CASE("rééchantillonnage : une table incohérente est refusée, jamais lue", "[resample]") {
  const auto good = ref::compute_weights(8, 4, 0.0, Filter::Spline36);
  REQUIRE(good);
  ref::Weights short_start = *good;
  short_start.start.pop_back();
  CHECK(ref::resample_rows(PlaneF(8, 2), short_start).error().code == ErrorCode::SizeMismatch);
  ref::Weights short_weights = *good;
  short_weights.w.resize(3);
  CHECK(ref::resample_columns(PlaneF(2, 8), short_weights).error().code == ErrorCode::SizeMismatch);
  ref::Weights no_taps = *good;
  no_taps.taps = 0;
  CHECK(ref::resample_rows(PlaneF(8, 2), no_taps).error().code == ErrorCode::SizeMismatch);
  // Le plan aussi : des dimensions sans les données qui vont avec.
  PlaneF hollow(8, 2);
  hollow.data.resize(4);
  CHECK(ref::resample_rows(hollow, *good).error().code == ErrorCode::SizeMismatch);
  CHECK(ref::resize(hollow, 4, 2, Filter::Spline36).error().code == ErrorCode::SizeMismatch);
}
