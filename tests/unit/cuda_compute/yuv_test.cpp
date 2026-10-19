// Ingestion YUV → RGB et sortie RGB → YUV / RGBA16 sur l'appareil, contre la référence CPU :
// exactes au bit (les .cu ne contractent pas en FMA), FP16 compris (arrondi au plus proche,
// égalités au pair, des deux côtés).
#include "gpu/cuda/compute/yuv.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "kernels/ref/color.h"
#include "kernels/ref/half.h"
#include "rig.h"

using namespace ajp;
using namespace ajp::gpu::cuda;
using namespace ajp::gpu::cuda::test;
using kernels::Filter;
using kernels::Format;
using kernels::Matrix;
using kernels::Range;
using kernels::Siting;

namespace {

constexpr int kWidth = 102;  // ni multiple de 32 ni de 8 : les bords des blocs sont exercés
constexpr int kHeight = 58;

constexpr Format kFormats[] = {Format::Nv12,    Format::P010,      Format::P016,
                               Format::Yuv420p, Format::Yuv420p10, Format::Yuv444p16};
constexpr Matrix kMatrices[] = {Matrix::Bt601, Matrix::Bt709, Matrix::Bt2020Ncl};
constexpr Range kRanges[] = {Range::Limited, Range::Full};
constexpr Siting kSitings[] = {Siting::Left, Siting::Center, Siting::TopLeft};
constexpr Filter kChromaFilters[] = {Filter::Spline36, Filter::Bilinear};

ColorSpec color(Matrix m, Range r, Siting s) {
  return {.matrix = m, .range = r, .siting = s};
}

kernels::ref::ColorSpec ref_color(const ColorSpec& c) {
  return {.matrix = c.matrix, .range = c.range, .siting = c.siting};
}

// RGB aléatoire hors gamut compris, plus quelques valeurs extrêmes : NaN, infinis, négatifs.
kernels::RgbImage awkward_rgb(int w, int h, std::uint32_t seed) {
  kernels::RgbImage image = random_rgb(w, h, -0.05F, 1.05F, seed);
  const float specials[] = {std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity(),
                            -1.0F,
                            2.0F,
                            0.5F};
  for (std::size_t i = 0; i < std::size(specials); ++i) image.planes[i % 3].data[i * 17] = specials[i];
  return image;
}

}  // namespace

TEST_CASE("l'ingestion YUV → RGB égale la référence au bit", "[cuda][yuv]") {
  AJP_CUDA_RIG(rig);
  const Format format = GENERATE(from_range(kFormats));
  const kernels::YuvImage image =
      random_yuv(format, kWidth, kHeight, 11U + static_cast<std::uint32_t>(format));
  const DeviceYuv src = upload_yuv(rig, image);
  DeviceTensor f32 = alloc_tensor(rig, Elem::F32, kWidth, kHeight);
  DeviceTensor f16 = alloc_tensor(rig, Elem::F16, kWidth, kHeight);
  for (const Matrix m : kMatrices) {
    for (const Range r : kRanges) {
      for (const Siting s : kSitings) {
        for (const Filter f : kChromaFilters) {
          CAPTURE(format, m, r, s, f);
          const ColorSpec c = color(m, r, s);
          auto plan = IngestPlan::create(
              *rig.pool, {.format = format, .width = kWidth, .height = kHeight, .color = c, .chroma = f},
              rig.s());
          REQUIRE(plan);
          REQUIRE(plan->run(src.planes, f32.tensor, rig.s()));
          REQUIRE(plan->run(src.planes, f16.tensor, rig.s()));
          const auto expected = kernels::ref::yuv_to_rgb(image, ref_color(c), f);
          REQUIRE(expected);
          CHECK(max_diff(download_rgb(rig, f32), *expected) == 0.0F);
          CHECK(max_diff(download_rgb(rig, f16), kernels::ref::round_to_half(*expected)) == 0.0F);
        }
      }
    }
  }
}

TEST_CASE("la sortie RGB → YUV égale la référence au code près", "[cuda][yuv]") {
  AJP_CUDA_RIG(rig);
  const Format format = GENERATE(from_range(kFormats));
  const kernels::RgbImage rgb = awkward_rgb(kWidth, kHeight, 23U + static_cast<std::uint32_t>(format));
  const DeviceTensor f32 = upload_rgb(rig, rgb, Elem::F32);
  const DeviceTensor f16 = upload_rgb(rig, rgb, Elem::F16);
  const kernels::RgbImage rgb16 = kernels::ref::round_to_half(rgb);
  DeviceYuv out = alloc_yuv(rig, format, kWidth, kHeight);
  for (const Matrix m : kMatrices) {
    for (const Range r : kRanges) {
      for (const Siting s : kSitings) {
        for (const Filter f : kChromaFilters) {
          CAPTURE(format, m, r, s, f);
          const ColorSpec c = color(m, r, s);
          auto plan = OutputPlan::create(
              *rig.pool, {.format = format, .width = kWidth, .height = kHeight, .color = c, .chroma = f},
              rig.s());
          REQUIRE(plan);
          REQUIRE(plan->run(f32.tensor, out.planes, rig.s()));
          const auto expected = kernels::ref::rgb_to_yuv(rgb, format, ref_color(c), f);
          REQUIRE(expected);
          CHECK(download_yuv(rig, out) == *expected);
          REQUIRE(plan->run(f16.tensor, out.planes, rig.s()));
          const auto expected16 = kernels::ref::rgb_to_yuv(rgb16, format, ref_color(c), f);
          REQUIRE(expected16);
          CHECK(download_yuv(rig, out) == *expected16);
        }
      }
    }
  }
}

TEST_CASE("l'aller-retour P010 de la chaîne est exact en luma", "[cuda][yuv]") {
  AJP_CUDA_RIG(rig);
  const kernels::YuvImage image = random_yuv(Format::P010, kWidth, kHeight, 5U);
  const DeviceYuv src = upload_yuv(rig, image);
  DeviceTensor rgb = alloc_tensor(rig, Elem::F32, kWidth, kHeight);
  DeviceYuv back = alloc_yuv(rig, Format::P010, kWidth, kHeight);
  const ColorSpec c = color(Matrix::Bt2020Ncl, Range::Limited, Siting::Left);
  auto in = IngestPlan::create(
      *rig.pool, {.format = Format::P010, .width = kWidth, .height = kHeight, .color = c}, rig.s());
  auto out = OutputPlan::create(
      *rig.pool, {.format = Format::P010, .width = kWidth, .height = kHeight, .color = c}, rig.s());
  REQUIRE(in);
  REQUIRE(out);
  REQUIRE(in->run(src.planes, rgb.tensor, rig.s()));
  REQUIRE(out->run(rgb.tensor, back.planes, rig.s()));
  CHECK(download_yuv(rig, back).y == image.y);
}

TEST_CASE("RGBA16 en mémoire linéaire égale to_rgba16 au bit", "[cuda][yuv]") {
  AJP_CUDA_RIG(rig);
  const kernels::RgbImage rgb = awkward_rgb(kWidth, kHeight, 31U);
  const Elem elem = GENERATE(Elem::F32, Elem::F16);
  const DeviceTensor t = upload_rgb(rig, rgb, elem);
  const std::size_t row = kWidth * 8;
  const std::size_t pitch = row + 256;
  auto buffer = rig.pool->allocate(pitch * kHeight, "test : RGBA16");
  REQUIRE(buffer);
  REQUIRE(write_rgba16(t.tensor, {.ptr = buffer->address(), .pitch = pitch}, rig.s()));
  const std::vector<std::byte> bytes = rig.download(buffer->data(), pitch * kHeight);
  std::vector<std::uint16_t> got;
  for (int y = 0; y < kHeight; ++y) {
    std::vector<std::uint16_t> line(kWidth * 4);
    std::memcpy(line.data(), &bytes[static_cast<std::size_t>(y) * pitch], row);
    got.insert(got.end(), line.begin(), line.end());
  }
  const auto expected = kernels::ref::to_rgba16(elem == Elem::F16 ? kernels::ref::round_to_half(rgb) : rgb);
  REQUIRE(expected);
  CHECK(got == *expected);
}

TEST_CASE("les plans YUV refusent une entrée hors contrat", "[cuda][yuv]") {
  AJP_CUDA_RIG(rig);
  const ColorSpec c = color(Matrix::Bt709, Range::Limited, Siting::Left);
  // Dimensions impaires en 4:2:0, ou nulles.
  for (const auto& [w, h] : {std::pair{101, 58}, std::pair{102, 0}, std::pair{0, 0}}) {
    auto plan =
        IngestPlan::create(*rig.pool, {.format = Format::Nv12, .width = w, .height = h, .color = c}, rig.s());
    REQUIRE_FALSE(plan);
    CHECK(plan.error().code == RefusalCode::ShapeInvalid);
  }
  // Area ne fait que réduire : la chroma 4:2:0 s'agrandit.
  auto area = IngestPlan::create(
      *rig.pool, {.format = Format::Nv12, .width = 64, .height = 64, .color = c, .chroma = Filter::Area},
      rig.s());
  REQUIRE_FALSE(area);
  CHECK(area.error().code == RefusalCode::ShapeInvalid);

  auto plan =
      IngestPlan::create(*rig.pool, {.format = Format::P010, .width = 64, .height = 32, .color = c}, rig.s());
  REQUIRE(plan);
  DeviceTensor rgb = alloc_tensor(rig, Elem::F16, 64, 32);
  DeviceYuv good = alloc_yuv(rig, Format::P010, 64, 32);
  REQUIRE(plan->run(good.planes, rgb.tensor, rig.s()));

  YuvPlanes short_pitch = good.planes;
  short_pitch.planes[1].pitch = 64;  // une ligne CbCr P010 de 64 pixels fait 128 octets
  CHECK(plan->run(short_pitch, rgb.tensor, rig.s()).error().code == RefusalCode::InputMismatch);
  YuvPlanes misaligned = good.planes;
  misaligned.planes[0].ptr += 1;
  CHECK(plan->run(misaligned, rgb.tensor, rig.s()).error().code == RefusalCode::InputMismatch);
  YuvPlanes other_format = good.planes;
  other_format.format = Format::P016;
  CHECK(plan->run(other_format, rgb.tensor, rig.s()).error().code == RefusalCode::ShapeInvalid);
  Tensor small = rgb.tensor;
  small.width = 32;
  CHECK(plan->run(good.planes, small, rig.s()).error().code == RefusalCode::ShapeInvalid);
  Tensor null = rgb.tensor;
  null.ptr = 0;
  CHECK(plan->run(good.planes, null, rig.s()).error().code == RefusalCode::InputMismatch);

  // Fenêtre de sortie hors du tenseur, ou de coin impair en 4:2:0.
  for (const kernels::Window w : {kernels::Window{.x = 2, .y = 0, .width = 64, .height = 32},
                                  kernels::Window{.x = 1, .y = 0, .width = 32, .height = 32}}) {
    auto out = OutputPlan::create(
        *rig.pool, {.format = Format::P010, .width = 64, .height = 32, .color = c, .window = w}, rig.s());
    REQUIRE_FALSE(out);
    CHECK(out.error().code == RefusalCode::ShapeInvalid);
  }
  // RGBA16 vers une ligne trop courte.
  CHECK(write_rgba16(rgb.tensor, {.ptr = good.planes.planes[0].ptr, .pitch = 64 * 4}, rig.s()).error().code ==
        RefusalCode::InputMismatch);
}
