// Parité au bit près entre la référence (src/kernels/ref/) et la transcription littérale des
// noyaux du moteur (engine_oracle.h, animejanai-inference-rtx@118c272), sur des images
// synthétiques aléatoires. Les deux calculent en float sans contraction FMA : tout écart est
// une erreur de reprise, pas un arrondi.
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "engine_oracle.h"
#include "kernels/ref/color.h"
#include "kernels/ref/detect.h"
#include "kernels/ref/half.h"
#include "kernels/ref/resample.h"
#include "kernels/ref/rife.h"
#include "kernels/ref/warp.h"
#include "kernels/spec.h"

using namespace ajp::kernels;

namespace {

int oracle_siting(Siting s) {
  return s == Siting::Left     ? oracle::SITING_LEFT
         : s == Siting::Center ? oracle::SITING_CENTER
                               : oracle::SITING_TOPLEFT;
}

int oracle_matrix(Matrix m) {
  return m == Matrix::Bt601   ? oracle::MATRIX_BT601
         : m == Matrix::Bt709 ? oracle::MATRIX_BT709
                              : oracle::MATRIX_BT2020;
}

int oracle_format(Format f) {
  return f == Format::Nv12 ? oracle::FMT_NV12 : f == Format::P010 ? oracle::FMT_P010 : oracle::FMT_YUV444P16;
}

oracle::csp oracle_csp(Format f, Matrix m, Range r) {
  return oracle::make_csp(oracle_format(f), oracle_matrix(m), r == Range::Full);
}

// Égalité au bit près : 0 et −0, ou deux NaN différents, ne passeraient pas pour égaux.
bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i])) return false;
  }
  return true;
}

// La mémoire du moteur : chroma Cb, Cr entrelacée, et tenseurs NCHW.
oracle::frame420 interleave(const YuvImage& in) {
  oracle::frame420 out{.w = in.width(), .h = in.height(), .y = in.y.data, .uv = {}};
  for (std::size_t i = 0; i < in.u.data.size(); ++i) {
    out.uv.push_back(in.u.data[i]);
    out.uv.push_back(in.v.data[i]);
  }
  return out;
}

std::vector<float> nchw(const RgbImage& in) {
  std::vector<float> out;
  for (const PlaneF& p : in.planes) out.insert(out.end(), p.data.begin(), p.data.end());
  return out;
}

YuvImage random_yuv(Format f, int w, int h, std::mt19937& rng) {
  std::uniform_int_distribution<std::uint32_t> pick(0, max_code(f) >> info(f).shift);
  YuvImage image(f, w, h, 0, 0, 0);
  for (PlaneU16* p : {&image.y, &image.u, &image.v}) {
    for (std::uint16_t& code : p->data) code = static_cast<std::uint16_t>(pick(rng) << info(f).shift);
  }
  return image;
}

// Un tenseur fp16 relu en float, hors gamut compris : ce que le moteur donne à la sortie.
RgbImage random_rgb(int w, int h, std::mt19937& rng) {
  std::uniform_real_distribution<float> pick(-0.1F, 1.1F);
  RgbImage image(w, h);
  for (PlaneF& p : image.planes) {
    for (float& v : p.data) v = ref::round_to_half(pick(rng));
  }
  return image;
}

}  // namespace

TEST_CASE("parité : tables de poids de compute() au bit près", "[parity][resample]") {
  const bool bilinear = GENERATE(false, true);
  for (const Siting s : {Siting::Left, Siting::Center, Siting::TopLeft}) {
    for (const int n : {1, 3, 7, 360, 683}) {
      const ChromaShift up = chroma_shift_up(s);
      const ChromaShift down = chroma_shift_down(s);
      for (const auto& [src, dst, shift] :
           {std::tuple{n, 2 * n, up.x}, std::tuple{n, 2 * n, up.y}, std::tuple{2 * n, n, down.x},
            std::tuple{2 * n, n, down.y}, std::tuple{3 * n, 2 * n, 0.0}, std::tuple{2 * n, 3 * n, 0.0}}) {
        CAPTURE(src, dst, shift, bilinear);
        const oracle::pass want = oracle::compute(src, dst, shift, bilinear);
        const auto got =
            ref::compute_weights(src, dst, shift, bilinear ? Filter::Bilinear : Filter::Spline36);
        REQUIRE(got);
        CHECK(got->taps == want.taps);
        CHECK(got->start == want.start);
        CHECK(same_bits(got->w, want.wt));
      }
    }
  }
}

TEST_CASE("parité : échelles et coefficients de make_csp", "[parity][color]") {
  for (const Format f : {Format::Nv12, Format::P010, Format::Yuv444p16}) {
    for (const Matrix m : {Matrix::Bt601, Matrix::Bt709, Matrix::Bt2020Ncl}) {
      for (const Range r : {Range::Limited, Range::Full}) {
        const oracle::csp want = oracle_csp(f, m, r);
        const YuvScale s = yuv_scale(f, r);
        const Coefficients k = coefficients(m);
        CHECK(s.yoff == want.yoff);
        CHECK(s.yscale == want.yscale);
        CHECK(s.coff == want.coff);
        CHECK(s.cscale == want.cscale);
        CHECK(k.kr == want.kr);
        CHECK(k.kb == want.kb);
      }
    }
  }
}

TEST_CASE("parité : ingestion et sortie 4:2:0, chaque siting et chaque filtre", "[parity][color]") {
  const Format format = GENERATE(Format::Nv12, Format::P010);
  const Siting siting = GENERATE(Siting::Left, Siting::Center, Siting::TopLeft);
  const Filter filter = GENERATE(Filter::Spline36, Filter::Bilinear);
  const Range range = GENERATE(Range::Limited, Range::Full);
  const Matrix matrix = GENERATE(Matrix::Bt601, Matrix::Bt2020Ncl);
  std::mt19937 rng(17);
  for (const auto& [w, h] : {std::pair{2, 2}, std::pair{6, 4}, std::pair{30, 18}, std::pair{64, 36}}) {
    CAPTURE(w, h);
    const ref::ColorSpec color{.matrix = matrix, .range = range, .siting = siting};
    const oracle::csp c = oracle_csp(format, matrix, range);
    const bool bilinear = filter == Filter::Bilinear;

    const YuvImage frame = random_yuv(format, w, h, rng);
    const auto rgb = ref::yuv_to_rgb(frame, color, filter);
    REQUIRE(rgb);
    CHECK(same_bits(nchw(*rgb), oracle::run_pre(interleave(frame), oracle_siting(siting), bilinear, c)));

    const RgbImage picture = random_rgb(w, h, rng);
    const auto yuv = ref::rgb_to_yuv(picture, format, color, filter);
    REQUIRE(yuv);
    const oracle::frame420 want =
        oracle::run_post(nchw(picture), w, h, format == Format::P010, oracle_siting(siting), bilinear, c);
    CHECK(yuv->y.data == want.y);
    CHECK(interleave(*yuv).uv == want.uv);
  }
}

TEST_CASE("parité : 4:4:4 16 bits, matrice seule", "[parity][color]") {
  const Range range = GENERATE(Range::Limited, Range::Full);
  std::mt19937 rng(23);
  const YuvImage frame = random_yuv(Format::Yuv444p16, 37, 21, rng);
  const ref::ColorSpec color{.matrix = Matrix::Bt709, .range = range};
  const oracle::csp c = oracle_csp(Format::Yuv444p16, Matrix::Bt709, range);
  const auto rgb = ref::yuv_to_rgb(frame, color);
  REQUIRE(rgb);
  CHECK(same_bits(nchw(*rgb),
                  oracle::run_pre444_pad(frame.y.data, frame.u.data, frame.v.data, 37, 21, 37, 21, 0, 0, c)));
  const RgbImage picture = random_rgb(37, 21, rng);
  const auto yuv = ref::rgb_to_yuv(picture, Format::Yuv444p16, color);
  REQUIRE(yuv);
  std::vector<std::uint16_t> got = yuv->y.data;
  got.insert(got.end(), yuv->u.data.begin(), yuv->u.data.end());
  got.insert(got.end(), yuv->v.data.begin(), yuv->v.data.end());
  CHECK(got == oracle::run_post444(nchw(picture), 37, 21, c));
}

TEST_CASE("parité : entrée RIFE remplie, préparation fusionnée du moteur", "[parity][rife]") {
  const Range range = GENERATE(Range::Limited, Range::Full);
  std::mt19937 rng(29);
  for (const auto& [w, h] : {std::pair{100, 60}, std::pair{130, 70}, std::pair{128, 64}}) {
    CAPTURE(w, h);
    const RifeGeometry g = rife_geometry(w, h);
    for (const Format f : {Format::Nv12, Format::P010}) {
      const YuvImage frame = random_yuv(f, w, h, rng);
      const auto got = ref::rife_prepare(frame, range);
      REQUIRE(got);
      CHECK(same_bits(nchw(*got),
                      oracle::run_rife_pre420(interleave(frame), g.padded_width, g.padded_height, g.left,
                                              g.top, f == Format::P010, oracle_csp(f, kRifeMatrix, range))));
    }
    const YuvImage frame = random_yuv(Format::Yuv444p16, w, h, rng);
    const auto got = ref::rife_prepare(frame, range);
    REQUIRE(got);
    CHECK(same_bits(nchw(*got), oracle::run_pre444_pad(frame.y.data, frame.u.data, frame.v.data, w, h,
                                                       g.padded_width, g.padded_height, g.left, g.top,
                                                       oracle_csp(Format::Yuv444p16, kRifeMatrix, range))));
    const auto constants = ref::rife_constant_planes(g.padded_width, g.padded_height);
    REQUIRE(constants);
    std::vector<float> planes;
    for (const PlaneF& p : *constants) planes.insert(planes.end(), p.data.begin(), p.data.end());
    CHECK(same_bits(planes, oracle::rife_consts(g.padded_width, g.padded_height)));
  }
}

TEST_CASE("parité : coupe de scène et dessins tenus", "[parity][detect]") {
  std::mt19937 rng(31);
  for (const Format f : {Format::Nv12, Format::P010, Format::Yuv444p16}) {
    for (const auto& [w, h] : {std::pair{32, 32}, std::pair{70, 40}, std::pair{200, 100}}) {
      CAPTURE(w, h);
      const YuvImage a = random_yuv(f, w, h, rng);
      YuvImage b = a;
      // Des écarts de quelques niveaux, ceux d'un dessin tenu sous le bruit de compression.
      std::uniform_int_distribution<int> delta(-3, 3);
      const auto unit = static_cast<int>(level_unit(f));
      for (std::uint16_t& code : b.y.data) {
        const int moved = int{code} + delta(rng) * unit;
        code = static_cast<std::uint16_t>(std::clamp(moved, 0, static_cast<int>(max_code(f))));
      }
      const auto worst = ref::block_diff_max(a.y, b.y, f);
      REQUIRE(worst);
      CHECK(*worst == oracle::block_diff(a.y.data, b.y.data, w, h, f == Format::Nv12 ? 1 : 2));
      // aji_scd_diff : 1/255 en NV12, 1/make_csp(plage pleine).yscale sinon.
      const float norm =
          f == Format::Nv12 ? 1.0F / 255.0F : 1.0F / oracle_csp(f, Matrix::Bt709, Range::Full).yscale;
      const auto sum = ref::scene_diff_sum(a.y, b.y, f);
      REQUIRE(sum);
      CHECK(*sum == oracle::scd_sum(a.y.data, b.y.data, norm));
    }
  }
}

TEST_CASE("parité : warp et mélange à tout instant", "[parity][warp]") {
  std::mt19937 rng(37);
  const int w = 48;
  const int h = 32;
  const Window area{.x = 4, .y = 2, .width = 24, .height = 16};
  const RgbImage a = random_rgb(w, h, rng);
  const RgbImage b = random_rgb(w, h, rng);
  ref::Motion motion{.flow = {}, .mask = PlaneF(64, 32), .area = area};
  std::uniform_real_distribution<float> displacement(-6.0F, 6.0F);
  std::uniform_real_distribution<float> weight(0.0F, 1.0F);
  std::vector<float> flow;
  for (PlaneF& p : motion.flow) {
    p = PlaneF(64, 32);
    for (float& v : p.data) v = ref::round_to_half(displacement(rng));
    flow.insert(flow.end(), p.data.begin(), p.data.end());
  }
  for (float& v : motion.mask.data) v = ref::round_to_half(weight(rng));
  for (const float t : {0.5F, 0.25F, 0.125F, 0.75F, 0.9F}) {
    CAPTURE(t);
    const auto got = ref::warp_blend_at(a, b, motion, t);
    REQUIRE(got);
    CHECK(same_bits(nchw(*got), oracle::warp_blend_mid(nchw(a), nchw(b), flow, motion.mask.data, w, h, 64, 32,
                                                       area.x, area.y, area.width, area.height, t)));
  }
  const auto middle = ref::warp_blend(a, b, motion);
  REQUIRE(middle);
  CHECK(same_bits(nchw(*middle), oracle::warp_blend_mid(nchw(a), nchw(b), flow, motion.mask.data, w, h, 64,
                                                        32, area.x, area.y, area.width, area.height, 0.5F)));
}
