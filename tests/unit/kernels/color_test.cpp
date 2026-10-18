#include "kernels/ref/color.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "kernels/spec.h"

using namespace ajp::kernels;
using Catch::Matchers::WithinAbs;

namespace {

struct Code {
  std::uint16_t y;
  std::uint16_t u;
  std::uint16_t v;
};

// Le code YUV d'une couleur RGB uniforme, par la référence (4:2:0 compris).
Code encode(Rgb color, Format f, Matrix m, Range r = Range::Limited) {
  RgbImage image(4, 4);
  for (int c = 0; c < 3; ++c) {
    const float value = c == 0 ? color.r : (c == 1 ? color.g : color.b);
    image.planes[static_cast<std::size_t>(c)] = PlaneF(4, 4, value);
  }
  const auto yuv = ref::rgb_to_yuv(image, f, {.matrix = m, .range = r});
  REQUIRE(yuv);
  return {.y = yuv->y.at(1, 1), .u = yuv->u.at(1, 1), .v = yuv->v.at(1, 1)};
}

Rgb decode(Code code, Format f, Matrix m, Range r = Range::Limited) {
  const auto rgb = ref::yuv_to_rgb(YuvImage(f, 4, 4, code.y, code.u, code.v), {.matrix = m, .range = r});
  REQUIRE(rgb);
  return rgb->at(1, 1);
}

void check_code(Code got, Code want) {
  CHECK(got.y == want.y);
  CHECK(got.u == want.u);
  CHECK(got.v == want.v);
}

// Un code aléatoire valide du format.
std::uint16_t random_code(std::mt19937& rng, Format f) {
  std::uniform_int_distribution<std::uint32_t> pick(0, max_code(f) >> info(f).shift);
  return static_cast<std::uint16_t>(pick(rng) << info(f).shift);
}

}  // namespace

TEST_CASE("couleur : échelles du conteneur identiques à make_csp du moteur", "[color]") {
  // aji_make_csp(format, matrice, plage) de resample.h, relevé pour chaque format du moteur.
  const auto same = [](YuvScale s, YuvScale want) {
    CHECK(s.yoff == want.yoff);
    CHECK(s.yscale == want.yscale);
    CHECK(s.coff == want.coff);
    CHECK(s.cscale == want.cscale);
  };
  same(yuv_scale(Format::Nv12, Range::Limited), {16, 219, 128, 224});
  same(yuv_scale(Format::Nv12, Range::Full), {0, 255, 128, 255});
  same(yuv_scale(Format::P010, Range::Limited), {4096, 56064, 32768, 57344});
  same(yuv_scale(Format::P010, Range::Full), {0, 65472, 32768, 65472});
  same(yuv_scale(Format::Yuv444p16, Range::Limited), {4096, 56064, 32768, 57344});
  same(yuv_scale(Format::Yuv444p16, Range::Full), {0, 65535, 32768, 65535});
  // Formats ajoutés : mêmes règles, profondeur et alignement propres.
  same(yuv_scale(Format::Yuv420p10, Range::Limited), {64, 876, 512, 896});
  same(yuv_scale(Format::Yuv420p10, Range::Full), {0, 1023, 512, 1023});
  same(yuv_scale(Format::P016, Range::Full), yuv_scale(Format::Yuv444p16, Range::Full));
  same(yuv_scale(Format::Yuv420p, Range::Limited), yuv_scale(Format::Nv12, Range::Limited));
}

TEST_CASE("couleur : quantification exacte de kernels.cu, égalités au pair", "[color]") {
  // P010 : 10 bits vrais puis décalage, qdiv 64 et qmax 1023.
  CHECK(quantizer(Format::P010).qdiv == 64.0F);
  CHECK(quantizer(Format::P010).qmax == 1023.0F);
  CHECK(quantize(32128.0F, Format::P010) == 32128);  // 502 × 64
  CHECK(quantize(32160.0F, Format::P010) == 32128);  // 502,5 → 502, pair
  CHECK(quantize(32224.0F, Format::P010) == 32256);  // 503,5 → 504, pair
  CHECK(quantize(32161.0F, Format::P010) == 32192);  // 502,52 → 503
  CHECK(quantize(70000.0F, Format::P010) == 65472);  // borné à 1023 × 64
  CHECK(quantize(127.5F, Format::Nv12) == 128);
  CHECK(quantize(126.5F, Format::Nv12) == 126);
  CHECK(quantize(-3.0F, Format::Nv12) == 0);
  CHECK(quantize(std::numeric_limits<float>::quiet_NaN(), Format::Nv12) == 0);
  CHECK(quantize(std::numeric_limits<float>::infinity(), Format::Nv12) == 255);
  CHECK(quantize(65535.4F, Format::Yuv444p16) == 65535);
  CHECK(quantize(1022.5F, Format::Yuv420p10) == 1022);
}

TEST_CASE("couleur : blanc, noir et gris 18 % en BT.709 limité", "[color]") {
  // 4:4:4 : exact. 4:2:0 : les poids float de la chroma somment à 1 près de 1e-7.
  const Rgb white = decode({235 * 256, 128 * 256, 128 * 256}, Format::Yuv444p16, Matrix::Bt709);
  CHECK(white.r == 1.0F);
  CHECK(white.g == 1.0F);
  CHECK(white.b == 1.0F);
  const Rgb black = decode({16, 128, 128}, Format::Nv12, Matrix::Bt709);
  CHECK_THAT(black.r, WithinAbs(0.0, 1e-6));
  CHECK_THAT(black.g, WithinAbs(0.0, 1e-6));
  CHECK_THAT(black.b, WithinAbs(0.0, 1e-6));
  const Rgb white420 = decode({235, 128, 128}, Format::Nv12, Matrix::Bt709);
  CHECK_THAT(white420.r, WithinAbs(1.0, 1e-6));
  CHECK_THAT(white420.b, WithinAbs(1.0, 1e-6));
  // Gris 0,18 : 16 + 0,18 × 219 = 55,42 → 55, chroma neutre.
  check_code(encode({0.18F, 0.18F, 0.18F}, Format::Nv12, Matrix::Bt709), {55, 128, 128});
  // En P010 : 64 + 0,18 × 876 = 221,68 → 222, décalé de 6 bits.
  check_code(encode({0.18F, 0.18F, 0.18F}, Format::P010, Matrix::Bt709), {222 * 64, 512 * 64, 512 * 64});
  // Plage pleine 8 et 16 bits.
  check_code(encode({1, 1, 1}, Format::Nv12, Matrix::Bt709, Range::Full), {255, 128, 128});
  check_code(encode({1, 1, 1}, Format::Yuv444p16, Matrix::Bt709, Range::Full), {65535, 32768, 32768});
  check_code(encode({0, 0, 0}, Format::Yuv444p16, Matrix::Bt709, Range::Full), {0, 32768, 32768});
}

TEST_CASE("couleur : primaires connues des trois matrices", "[color]") {
  // BT.709 8 bits : rouge (63, 102, 240), vert (173, 42, 26), bleu (32, 240, 118).
  check_code(encode({1, 0, 0}, Format::Nv12, Matrix::Bt709), {63, 102, 240});
  check_code(encode({0, 1, 0}, Format::Nv12, Matrix::Bt709), {173, 42, 26});
  check_code(encode({0, 0, 1}, Format::Yuv420p, Matrix::Bt709), {32, 240, 118});
  // BT.601 8 bits : rouge (81, 90, 240), vert (145, 54, 34), bleu (41, 240, 110).
  check_code(encode({1, 0, 0}, Format::Nv12, Matrix::Bt601), {81, 90, 240});
  check_code(encode({0, 1, 0}, Format::Nv12, Matrix::Bt601), {145, 54, 34});
  check_code(encode({0, 0, 1}, Format::Nv12, Matrix::Bt601), {41, 240, 110});
  // BT.709 10 bits : rouge (250, 409, 960), en P010 décalé de 6 bits.
  check_code(encode({1, 0, 0}, Format::P010, Matrix::Bt709), {250 * 64, 409 * 64, 960 * 64});
  // BT.2020 10 bits : Y = 64 + 876 × 0,2627 = 294,13 ; Cb = 512 − 896 × 0,2627 / 1,8814 = 386,89.
  check_code(encode({1, 0, 0}, Format::Yuv420p10, Matrix::Bt2020Ncl), {294, 387, 960});
  // Et le décodage retrouve la primaire à la quantification près.
  const Rgb red = decode({63, 102, 240}, Format::Nv12, Matrix::Bt709);
  CHECK_THAT(red.r, WithinAbs(1.0, 0.01));
  CHECK_THAT(red.g, WithinAbs(0.0, 0.01));
  CHECK_THAT(red.b, WithinAbs(0.0, 0.01));
}

TEST_CASE("couleur : ingestion NV12 calculée à la main, Spline36 en siting gauche", "[color][parity]") {
  // 4 × 2, chroma 2 × 1 : Cb = (100, 200), Cr neutre, Y = 126.
  // Poids horizontaux de compute_filter (pos = (x + 0,5) / 2 + 0,25, 6 prises, miroir) :
  //   x = 0 : (0, 0, 1, 0, 0, 0) sur c0 → 100 ; x = 2 : de même sur c1 → 200 ;
  //   x = 1 : (3, −18, 91, 91, −18, 3) / 152 → (c0 + c1) / 2 = 150 ;
  //   x = 3 : −30/152 c0 + 182/152 c1 = 33400/152 = 219,7368 (dépassement, jamais borné).
  // Une seule ligne de chroma : la passe verticale la recopie.
  YuvImage frame(Format::Nv12, 4, 2, 126, 128, 128);
  frame.u.at(0, 0) = 100;
  frame.u.at(1, 0) = 200;
  const auto rgb =
      ref::yuv_to_rgb(frame, {.matrix = Matrix::Bt709, .range = Range::Limited, .siting = Siting::Left});
  REQUIRE(rgb);
  const float y = (126.0F - 16.0F) / 219.0F;
  const float kb2 = 2.0F * (1.0F - 0.0722F);
  const float cb[4] = {100.0F, 150.0F, 200.0F, 33400.0F / 152.0F};
  for (int x = 0; x < 4; ++x) {
    const float u = (cb[x] - 128.0F) / 224.0F;
    CHECK_THAT(rgb->planes[2].at(x, 0), WithinAbs(y + kb2 * u, 1e-5));
    CHECK_THAT(rgb->planes[2].at(x, 1), WithinAbs(y + kb2 * u, 1e-5));
    CHECK_THAT(rgb->planes[0].at(x, 0), WithinAbs(y, 1e-6));  // Cr neutre : R = Y
  }
  CHECK(rgb->planes[2].at(3, 0) > 1.0F);  // hors gamut, passé tel quel au réseau
}

TEST_CASE("couleur : le siting déplace la chroma comme dans zimg", "[color]") {
  // Centre : décalage nul, la luma 0 tombe entre deux échantillons et ne recopie plus c0.
  YuvImage frame(Format::Nv12, 8, 8, 126, 128, 128);
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) frame.u.at(x, y) = static_cast<std::uint16_t>(60 + 40 * x + 10 * y);
  }
  const auto left = ref::yuv_to_rgb(frame, {.siting = Siting::Left});
  const auto center = ref::yuv_to_rgb(frame, {.siting = Siting::Center});
  const auto top_left = ref::yuv_to_rgb(frame, {.siting = Siting::TopLeft});
  REQUIRE(left);
  REQUIRE(center);
  REQUIRE(top_left);
  // Haut-gauche : la luma (2, 2) est co-située avec c(1, 1) = 110 dans les deux directions.
  const float b11 = (126.0F - 16.0F) / 219.0F + 2.0F * (1.0F - 0.0722F) * (110.0F - 128.0F) / 224.0F;
  CHECK_THAT(top_left->planes[2].at(2, 2), WithinAbs(b11, 1e-5));
  // Gauche : co-située en x seulement (pos y = 1,25) ; centre : dans aucune direction.
  CHECK(std::abs(left->planes[2].at(2, 2) - b11) > 1e-3F);
  CHECK(std::abs(center->planes[2].at(2, 2) - left->planes[2].at(2, 2)) > 1e-3F);
  CHECK(chroma_shift_up(Siting::Left).x == 0.25);
  CHECK(chroma_shift_up(Siting::TopLeft).y == 0.25);
  CHECK(chroma_shift_down(Siting::Center).x == 0.0);
  CHECK(chroma_shift_down(Siting::Left).x == -0.5);
}

TEST_CASE("couleur : aller-retour 4:4:4 exact, toutes profondeurs et plages", "[color][roundtrip]") {
  const Matrix matrix = GENERATE(Matrix::Bt601, Matrix::Bt709, Matrix::Bt2020Ncl);
  const Range range = GENERATE(Range::Limited, Range::Full);
  std::mt19937 rng(7);
  YuvImage frame(Format::Yuv444p16, 16, 8, 0, 0, 0);
  for (PlaneU16* p : {&frame.y, &frame.u, &frame.v}) {
    for (std::uint16_t& code : p->data) code = random_code(rng, Format::Yuv444p16);
  }
  const ref::ColorSpec color{.matrix = matrix, .range = range};
  const auto back = ref::yuv_to_rgb(frame, color).and_then([&color](const RgbImage& rgb) {
    return ref::rgb_to_yuv(rgb, Format::Yuv444p16, color);
  });
  REQUIRE(back);
  CHECK(*back == frame);
  static_assert(tolerance::kRoundTrip == 0);
}

TEST_CASE("couleur : aller-retour 4:2:0 exact en luma, et en chroma uniforme", "[color][roundtrip]") {
  const Format format =
      GENERATE(Format::Nv12, Format::P010, Format::P016, Format::Yuv420p, Format::Yuv420p10);
  const Siting siting = GENERATE(Siting::Left, Siting::Center, Siting::TopLeft);
  std::mt19937 rng(11);
  YuvImage frame(format, 16, 12, 0, random_code(rng, format), random_code(rng, format));
  for (std::uint16_t& code : frame.y.data) code = random_code(rng, format);
  const ref::ColorSpec color{.matrix = Matrix::Bt709, .range = Range::Limited, .siting = siting};
  const auto back = ref::yuv_to_rgb(frame, color).and_then([&](const RgbImage& rgb) {
    return ref::rgb_to_yuv(rgb, format, color);
  });
  REQUIRE(back);
  CHECK(*back == frame);
}

TEST_CASE("couleur : RGBA16 UNORM, clamp puis arrondi", "[color]") {
  CHECK(unorm16(0.0F) == 0);
  CHECK(unorm16(1.0F) == 65535);
  CHECK(unorm16(0.5F) == 32768);  // 32767,5 + 0,5 tronqué
  CHECK(unorm16(1.0F / 65535.0F) == 1);
  CHECK(unorm16(-0.25F) == 0);
  CHECK(unorm16(1.5F) == 65535);
  CHECK(unorm16(std::numeric_limits<float>::quiet_NaN()) == 0);
  RgbImage image(2, 1, 0.0F);
  image.planes[0].at(1, 0) = 1.0F;
  image.planes[2].at(0, 0) = 0.25F;
  const auto rgba = ref::to_rgba16(image);
  REQUIRE(rgba);
  CHECK(*rgba == std::vector<std::uint16_t>{0, 0, 16384, 65535, 65535, 0, 0, 65535});
}

TEST_CASE("couleur : étiquettes manquantes, règles du fork", "[color]") {
  CHECK(effective_matrix(std::nullopt, 576) == Matrix::Bt601);
  CHECK(effective_matrix(std::nullopt, 719) == Matrix::Bt601);
  CHECK(effective_matrix(std::nullopt, 720) == Matrix::Bt709);
  CHECK(effective_matrix(Matrix::Bt2020Ncl, 480) == Matrix::Bt2020Ncl);
  CHECK(effective_range(std::nullopt) == Range::Limited);
  CHECK(effective_range(Range::Full) == Range::Full);
  CHECK(effective_siting(std::nullopt) == Siting::Left);
  CHECK(effective_siting(Siting::TopLeft) == Siting::TopLeft);
  CHECK(kOutputSiting == Siting::Left);
}

TEST_CASE("couleur : une image hors contrat est refusée", "[color]") {
  YuvImage p010(Format::P010, 4, 4, 4096, 32768, 32768);
  p010.y.at(0, 0) = 4097;  // bits bas non nuls
  CHECK(ref::check_yuv(p010).error().code == ErrorCode::SampleOutOfRange);
  YuvImage nv12(Format::Nv12, 4, 4, 16, 128, 128);
  nv12.u.at(0, 0) = 256;
  CHECK(ref::check_yuv(nv12).error().code == ErrorCode::SampleOutOfRange);
  CHECK(ref::check_yuv(YuvImage(Format::Nv12, 5, 4, 16, 128, 128)).error().code == ErrorCode::OddDimension);
  YuvImage wrong(Format::Nv12, 4, 4, 16, 128, 128);
  wrong.v = PlaneU16(4, 4, 128);
  CHECK(ref::check_yuv(wrong).error().code == ErrorCode::SizeMismatch);
  CHECK(ref::check_yuv(YuvImage(Format::Yuv444p16, 3, 5, 0, 0, 0)).has_value());  // 4:4:4 impair admis
  CHECK(ref::rgb_to_yuv(RgbImage(6, 3), Format::Nv12, {}).error().code == ErrorCode::OddDimension);
}
