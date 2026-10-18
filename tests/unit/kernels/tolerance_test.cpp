// Les bornes des ports (spec/tolerance.h) contre un port simulé : chaque a × b + c contracté
// en FMA, comme nvcc et la plupart des compilateurs de shaders le font, et les tenseurs rangés
// en fp16. Le port reçoit les mêmes entrées que la référence ; seuls ses arrondis diffèrent.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <random>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "kernels/ref/color.h"
#include "kernels/ref/half.h"
#include "kernels/ref/resample.h"
#include "kernels/ref/warp.h"
#include "kernels/spec.h"

using namespace ajp::kernels;

namespace {

YuvImage random_yuv(Format f, int w, int h, std::mt19937& rng) {
  std::uniform_int_distribution<std::uint32_t> pick(0, max_code(f) >> info(f).shift);
  YuvImage image(f, w, h, 0, 0, 0);
  for (PlaneU16* p : {&image.y, &image.u, &image.v}) {
    for (std::uint16_t& code : p->data) code = static_cast<std::uint16_t>(pick(rng) << info(f).shift);
  }
  return image;
}

RgbImage random_rgb(int w, int h, std::mt19937& rng) {
  std::uniform_real_distribution<float> pick(-0.05F, 1.05F);
  RgbImage image(w, h);
  for (PlaneF& p : image.planes) {
    for (float& v : p.data) v = pick(rng);
  }
  return image;
}

// Les deux passes séparables, accumulées en FMA.
PlaneF fma_resample(const PlaneF& in, int width, int height, ChromaShift shift, Filter filter) {
  const auto wx = ref::compute_weights(in.width, width, shift.x, filter);
  const auto wy = ref::compute_weights(in.height, height, shift.y, filter);
  REQUIRE(wx);
  REQUIRE(wy);
  PlaneF rows(width, in.height);
  for (int y = 0; y < in.height; ++y) {
    for (int x = 0; x < width; ++x) {
      float a = 0.0F;
      for (int j = 0; j < wx->taps; ++j) {
        a = std::fma(wx->row(x)[static_cast<std::size_t>(j)],
                     in.at(mirror_index(wx->start[static_cast<std::size_t>(x)] + j, in.width), y), a);
      }
      rows.at(x, y) = a;
    }
  }
  PlaneF out(width, height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float a = 0.0F;
      for (int j = 0; j < wy->taps; ++j) {
        a = std::fma(wy->row(y)[static_cast<std::size_t>(j)],
                     rows.at(x, mirror_index(wy->start[static_cast<std::size_t>(y)] + j, in.height)), a);
      }
      out.at(x, y) = a;
    }
  }
  return out;
}

PlaneF to_float(const PlaneU16& p) {
  PlaneF out(p.width, p.height);
  for (std::size_t i = 0; i < p.data.size(); ++i) out.data[i] = static_cast<float>(p.data[i]);
  return out;
}

// Ingestion d'un port FMA, rangée en fp16.
RgbImage fma_yuv_to_rgb(const YuvImage& in, const ref::ColorSpec& color) {
  const int w = in.width();
  const int h = in.height();
  PlaneF u = to_float(in.u);
  PlaneF v = to_float(in.v);
  if (info(in.format).subsampled) {
    u = fma_resample(u, w, h, chroma_shift_up(color.siting), kChainChromaFilter);
    v = fma_resample(v, w, h, chroma_shift_up(color.siting), kChainChromaFilter);
  }
  const YuvScale s = yuv_scale(in.format, color.range);
  const Coefficients k = coefficients(color.matrix);
  const float kg = 1.0F - k.kr - k.kb;
  RgbImage out(w, h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const Yuv p = normalize(static_cast<float>(in.y.at(x, y)), u.at(x, y), v.at(x, y), s);
      const float g =
          p.y - std::fma(2.0F * k.kb * (1.0F - k.kb), p.u, 2.0F * k.kr * (1.0F - k.kr) * p.v) / kg;
      out.planes[0].at(x, y) = ref::round_to_half(std::fma(2.0F * (1.0F - k.kr), p.v, p.y));
      out.planes[1].at(x, y) = ref::round_to_half(g);
      out.planes[2].at(x, y) = ref::round_to_half(std::fma(2.0F * (1.0F - k.kb), p.u, p.y));
    }
  }
  return out;
}

// Sortie quantifiée d'un port FMA.
YuvImage fma_rgb_to_yuv(const RgbImage& in, Format format, const ref::ColorSpec& color) {
  const int w = in.width();
  const int h = in.height();
  const YuvScale s = yuv_scale(format, color.range);
  const Coefficients k = coefficients(color.matrix);
  YuvImage out(format, w, h, 0, 0, 0);
  PlaneF u(w, h);
  PlaneF v(w, h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const Rgb p = in.at(x, y);
      const float luma = std::fma(k.kb, p.b, std::fma(1.0F - k.kr - k.kb, p.g, k.kr * p.r));
      out.y.at(x, y) = quantize(std::fma(luma, s.yscale, s.yoff), format);
      u.at(x, y) = (p.b - luma) / (2.0F * (1.0F - k.kb));
      v.at(x, y) = (p.r - luma) / (2.0F * (1.0F - k.kr));
    }
  }
  if (info(format).subsampled) {
    u = fma_resample(u, w / 2, h / 2, chroma_shift_down(color.siting), kChainChromaFilter);
    v = fma_resample(v, w / 2, h / 2, chroma_shift_down(color.siting), kChainChromaFilter);
  }
  for (std::size_t i = 0; i < u.data.size(); ++i) {
    out.u.data[i] = quantize(std::fma(u.data[i], s.cscale, s.coff), format);
    out.v.data[i] = quantize(std::fma(v.data[i], s.cscale, s.coff), format);
  }
  return out;
}

// Écart maximal entre deux plans de codes, en codes du format (P010 : en pas de 10 bits).
int max_code_gap(const PlaneU16& a, const PlaneU16& b, Format f) {
  int gap = 0;
  for (std::size_t i = 0; i < a.data.size(); ++i) {
    gap = std::max(gap, std::abs(int{a.data[i]} - int{b.data[i]}) >> info(f).shift);
  }
  return gap;
}

}  // namespace

TEST_CASE("tolérance : un port FMA rangé en fp16 reste dans kRgbF16", "[tolerance]") {
  const Format format = GENERATE(Format::Nv12, Format::P010, Format::Yuv444p16);
  const Range range = GENERATE(Range::Limited, Range::Full);
  std::mt19937 rng(41);
  const YuvImage frame = random_yuv(format, 64, 36, rng);
  const ref::ColorSpec color{.matrix = Matrix::Bt709, .range = range, .siting = Siting::Left};
  const auto want = ref::yuv_to_rgb(frame, color);
  REQUIRE(want);
  const RgbImage port = fma_yuv_to_rgb(frame, color);
  float gap = 0.0F;
  for (std::size_t c = 0; c < 3; ++c) {
    for (std::size_t i = 0; i < port.planes[c].data.size(); ++i) {
      gap = std::max(gap, std::fabs(port.planes[c].data[i] - want->planes[c].data[i]));
    }
  }
  CHECK(gap <= tolerance::kRgbF16);
  CHECK(gap > 0.0F);  // l'arrondi fp16 se voit : la borne n'est pas vide
}

TEST_CASE("tolérance : la sortie d'un port FMA reste à un code de la référence", "[tolerance]") {
  const Format format = GENERATE(Format::Nv12, Format::P010, Format::P016, Format::Yuv444p16);
  const Matrix matrix = GENERATE(Matrix::Bt709, Matrix::Bt2020Ncl);
  std::mt19937 rng(43);
  const RgbImage picture = random_rgb(64, 36, rng);
  const ref::ColorSpec color{.matrix = matrix, .range = Range::Limited, .siting = kOutputSiting};
  const auto want = ref::rgb_to_yuv(picture, format, color);
  REQUIRE(want);
  const YuvImage port = fma_rgb_to_yuv(picture, format, color);
  CHECK(max_code_gap(port.y, want->y, format) <= tolerance::kCode);
  CHECK(max_code_gap(port.u, want->u, format) <= tolerance::kCode);
  CHECK(max_code_gap(port.v, want->v, format) <= tolerance::kCode);
}

TEST_CASE("tolérance : le warp d'un port FMA rangé en fp16 reste dans kWarp", "[tolerance][warp]") {
  // Entrées fp16 communes ; le port contracte la lecture bilinéaire et le mélange, puis range
  // sa sortie en fp16. Instant estimé (t = 1/2), la branche sans logf ni expf.
  std::mt19937 rng(47);
  const int w = 40;
  const int h = 24;
  RgbImage a = ref::round_to_half(random_rgb(w, h, rng));
  RgbImage b = ref::round_to_half(random_rgb(w, h, rng));
  std::uniform_real_distribution<float> displacement(-4.0F, 4.0F);
  std::uniform_real_distribution<float> weight(0.0F, 1.0F);
  ref::Motion motion{.flow = {}, .mask = PlaneF(w / 2, h / 2), .area = {0, 0, w / 2, h / 2}};
  for (PlaneF& p : motion.flow) {
    p = PlaneF(w / 2, h / 2);
    for (float& v : p.data) v = ref::round_to_half(displacement(rng));
  }
  for (float& v : motion.mask.data) v = ref::round_to_half(weight(rng));
  const auto want = ref::warp_blend(a, b, motion);
  REQUIRE(want);

  const auto fma_sample = [](const PlaneF& p, float x, float y) {
    x = std::fmin(std::fmax(x, 0.0F), static_cast<float>(p.width - 1));
    y = std::fmin(std::fmax(y, 0.0F), static_cast<float>(p.height - 1));
    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(y);
    const int x1 = std::min(x0 + 1, p.width - 1);
    const int y1 = std::min(y0 + 1, p.height - 1);
    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);
    const float top = std::fma(p.at(x1, y0), fx, p.at(x0, y0) * (1.0F - fx));
    const float bottom = std::fma(p.at(x1, y1), fx, p.at(x0, y1) * (1.0F - fx));
    return std::fma(bottom, fy, top * (1.0F - fy));
  };
  float gap = 0.0F;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      // Image deux fois plus grande que le mouvement : sx = sy = 2.
      const float u = (static_cast<float>(x) + 0.5F) / 2.0F - 0.5F;
      const float v = (static_cast<float>(y) + 0.5F) / 2.0F - 0.5F;
      const float m = fma_sample(motion.mask, u, v);
      const float ax = std::fma(2.0F, fma_sample(motion.flow[0], u, v), static_cast<float>(x));
      const float ay = std::fma(2.0F, fma_sample(motion.flow[1], u, v), static_cast<float>(y));
      const float bx = std::fma(2.0F, fma_sample(motion.flow[2], u, v), static_cast<float>(x));
      const float by = std::fma(2.0F, fma_sample(motion.flow[3], u, v), static_cast<float>(y));
      for (std::size_t c = 0; c < 3; ++c) {
        const float va = fma_sample(a.planes[c], ax, ay);
        const float vb = fma_sample(b.planes[c], bx, by);
        const float port = ref::round_to_half(std::fma(va, m, vb * (1.0F - m)));
        gap = std::max(gap, std::fabs(port - want->planes[c].at(x, y)));
      }
    }
  }
  CHECK(gap <= tolerance::kWarp);
}
