// L'ingestion du backend TensorRT contre la référence CPU, chroma comprise (B01-03, B03-02) : le 4:2:0 de
// NVDEC au filtre demandé (Spline36 de la chaîne, bilinéaire) et à chaque siting, le 4:4:4 et le 4:2:2 du
// décodage logiciel sans réduction de la chroma. Sans GPU NVIDIA, chaque test se SKIP.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "backend_fixture.h"
#include "backend_frames.h"
#include "gpu/backend.h"
#include "kernels/ref/color.h"
#include "kernels/ref/resample.h"

namespace {

using namespace ajp;
using namespace ajp::test;

// Écart le plus grand entre le tenseur ingéré et la référence, sur toute l'image, bords compris.
double worst_gap(const std::vector<float>& got, const kernels::RgbImage& want, int w, int h) {
  double worst = 0.0;
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const float have = got[static_cast<std::size_t>((c * h + y) * w + x)];
        const float expected = want.planes[static_cast<std::size_t>(c)].at(x, y);
        worst = std::max(worst, static_cast<double>(std::fabs(expected - have)));
      }
    }
  }
  return worst;
}

kernels::Siting siting_of(ChromaSiting siting) {
  return siting == ChromaSiting::Center    ? kernels::Siting::Center
         : siting == ChromaSiting::TopLeft ? kernels::Siting::TopLeft
                                           : kernels::Siting::Left;
}

}  // namespace

TEST_CASE("Backend TensorRT : l'ingestion suit le filtre de chroma et le siting (B03-02)",
          "[backend_tensorrt][cuda]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  const kernels::Filter filter = GENERATE(kernels::Filter::Spline36, kernels::Filter::Bilinear);
  const ChromaSiting siting = GENERATE(ChromaSiting::Left, ChromaSiting::Center, ChromaSiting::TopLeft);
  CAPTURE(static_cast<int>(filter), static_cast<int>(siting));
  constexpr int w = 64;
  constexpr int h = 36;
  // Chroma en damier d'un point : le cas où bilinéaire et Spline36 s'écartent le plus.
  kernels::YuvImage image(kernels::Format::Nv12, w, h, 16, 128, 128);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x)
      image.y.data[image.y.index(x, y)] = static_cast<std::uint16_t>(40 + (x * 5 + y * 3) % 180);
  }
  for (int y = 0; y < h / 2; ++y) {
    for (int x = 0; x < w / 2; ++x) {
      const bool odd = (x + y) % 2 == 1;
      image.u.data[image.u.index(x, y)] = static_cast<std::uint16_t>(odd ? 200 : 60);
      image.v.data[image.v.index(x, y)] = static_cast<std::uint16_t>(odd ? 70 : 190);
    }
  }
  const auto expected = kernels::ref::yuv_to_rgb(
      image, {.matrix = kernels::Matrix::Bt709, .siting = siting_of(siting)}, filter);
  REQUIRE(expected);

  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  auto rgb = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, h, w}});
  REQUIRE(rgb);
  DecodedFrame frame = frame_of(std::make_shared<CudaNv12>(image), w, h);
  frame.color = ColorInfo{.matrix = ColorMatrix::Bt709, .range = ColorRange::Limited, .siting = siting};
  REQUIRE(loaded.backend->kernels()->ingest(frame, gpu::IngestSpec{.chroma = filter}, (*rgb)->tensor()));
  CHECK(worst_gap(download((*rgb)->tensor()), *expected, w, h) < 2e-3);
  // Area ne fait que réduire : refusé, jamais ignoré.
  CHECK(loaded.backend->kernels()
            ->ingest(frame, gpu::IngestSpec{.chroma = kernels::Filter::Area}, (*rgb)->tensor())
            .error()
            .code == RefusalCode::FormatUnsupported);
}

TEST_CASE("Backend TensorRT : le 4:4:4 du décodage logiciel garde sa chroma entière (B01-03)",
          "[backend_tensorrt][cuda][host]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  constexpr int w = 70;
  constexpr int h = 38;
  const auto unit = static_cast<std::uint16_t>(kernels::level_unit(kernels::Format::Yuv444p16));
  kernels::YuvImage image(kernels::Format::Yuv444p16, w, h, static_cast<std::uint16_t>(16 * unit),
                          static_cast<std::uint16_t>(128 * unit), static_cast<std::uint16_t>(128 * unit));
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      image.y.data[image.y.index(x, y)] = static_cast<std::uint16_t>((16 + (x * 3 + y * 5) % 220) * unit);
      // Une chroma qui change à chaque point : une réduction en 4:2:0 la ferait baver.
      image.u.data[image.u.index(x, y)] = static_cast<std::uint16_t>(((x % 2 == 0) ? 60 : 196) * unit);
      image.v.data[image.v.index(x, y)] = static_cast<std::uint16_t>(((y % 2 == 0) ? 80 : 176) * unit);
    }
  }
  const auto expected = kernels::ref::yuv_to_rgb(image, {.matrix = kernels::Matrix::Bt709});
  REQUIRE(expected);

  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  auto rgb = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, h, w}});
  REQUIRE(rgb);
  DecodedFrame frame = frame_of(std::make_shared<HostYuv>(image), w, h);
  frame.format = PixelFormat::Yuv444p16;
  frame.color = ColorInfo{.matrix = ColorMatrix::Bt709, .range = ColorRange::Limited};
  for (int i = 0; i < 4; ++i) REQUIRE(loaded.backend->kernels()->ingest(frame, {}, (*rgb)->tensor()));
  CHECK(worst_gap(download((*rgb)->tensor()), *expected, w, h) < 2e-3);
}

TEST_CASE("Backend TensorRT : le 4:2:2 du décodage logiciel n'est suréchantillonné qu'en largeur (B01-03)",
          "[backend_tensorrt][cuda][host]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  const ChromaSiting siting = GENERATE(ChromaSiting::Left, ChromaSiting::Center);
  CAPTURE(static_cast<int>(siting));
  constexpr int w = 70;
  constexpr int h = 38;
  constexpr int cw = w / 2;
  const auto unit = static_cast<std::uint16_t>(kernels::level_unit(kernels::Format::Yuv444p16));
  kernels::PlaneU16 luma(w, h);
  kernels::PlaneU16 cb(cw, h);
  kernels::PlaneU16 cr(cw, h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x)
      luma.at(x, y) = static_cast<std::uint16_t>((16 + (x * 3 + y * 5) % 220) * unit);
    for (int x = 0; x < cw; ++x) {
      // Une ligne sur deux change : en 4:2:0, la hauteur aussi serait réduite.
      cb.at(x, y) = static_cast<std::uint16_t>(((x + y) % 2 == 0 ? 60 : 196) * unit);
      cr.at(x, y) = static_cast<std::uint16_t>((y % 2 == 0 ? 80 : 176) * unit);
    }
  }
  // Référence : la passe horizontale de kernels/ref/resample.h en Spline36, puis la matrice point par point.
  const kernels::ChromaShift shift = kernels::chroma_shift_up(siting_of(siting));
  const auto wx = kernels::ref::compute_weights(cw, w, shift.x, kernels::Filter::Spline36);
  REQUIRE(wx);
  kernels::PlaneF cb_float(cw, h);
  kernels::PlaneF cr_float(cw, h);
  for (std::size_t i = 0; i < cb.data.size(); ++i) {
    cb_float.data[i] = static_cast<float>(cb.data[i]);
    cr_float.data[i] = static_cast<float>(cr.data[i]);
  }
  const auto cb_full = kernels::ref::resample_rows(cb_float, *wx);
  const auto cr_full = kernels::ref::resample_rows(cr_float, *wx);
  REQUIRE(cb_full);
  REQUIRE(cr_full);
  const kernels::YuvScale scale = kernels::yuv_scale(kernels::Format::Yuv444p16, kernels::Range::Limited);
  const kernels::Coefficients k = kernels::coefficients(kernels::Matrix::Bt709);
  kernels::RgbImage expected(w, h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const kernels::Rgb p = kernels::to_rgb(
          kernels::normalize(static_cast<float>(luma.at(x, y)), cb_full->at(x, y), cr_full->at(x, y), scale),
          k);
      expected.planes[0].at(x, y) = p.r;
      expected.planes[1].at(x, y) = p.g;
      expected.planes[2].at(x, y) = p.b;
    }
  }

  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  auto rgb = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, h, w}});
  REQUIRE(rgb);
  DecodedFrame frame = frame_of(std::make_shared<HostYuv>(luma, cb, cr), w, h);
  frame.format = PixelFormat::Yuv422p16;
  frame.color = ColorInfo{.matrix = ColorMatrix::Bt709, .range = ColorRange::Limited, .siting = siting};
  for (int i = 0; i < 4; ++i) REQUIRE(loaded.backend->kernels()->ingest(frame, {}, (*rgb)->tensor()));
  CHECK(worst_gap(download((*rgb)->tensor()), expected, w, h) < 2e-3);
}
