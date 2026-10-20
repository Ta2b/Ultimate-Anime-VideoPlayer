// Extensions de la tranche verticale (gpu/kernel_ops.h) sur le backend Vulkan, contre les mêmes
// références que le backend TensorRT : réduction fusionnée à l'ingestion, entrée de RIFE depuis
// le RGB, mesures de la paire, plus grand déplacement, warp à poses gardées.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "backend_rig.h"
#include "gpu/vulkan/backend/reduce_weights.h"
#include "kernels/ref/resample.h"

using namespace ajp;
using namespace ajp::gpu;
using namespace ajp::gpu::vk::test;
namespace k = ajp::kernels;

namespace {

k::RgbImage download(vk::VulkanBackend& backend, const TensorBuffer& t, int w, int h) {
  const std::size_t element = t.tensor().type == TensorType::F16 ? 2 : 4;
  return rgb_from(read_tensor(backend, t.tensor(), element * 3 * static_cast<std::size_t>(w * h)), w, h,
                  t.tensor().type);
}

// Noyau de la réduction filtrée (slice_kernels.cu) : triangle, Spline36 ou cubique B, C.
double reduce_weight(const Reduction& r, double d) {
  d = std::fabs(d);
  if (r.kind == ReduceKind::Triangle) return std::max(0.0, 1.0 - d);
  if (r.kind == ReduceKind::Spline36) {
    if (d < 1.0) return ((13.0 / 11.0 * d - 453.0 / 209.0) * d - 3.0 / 209.0) * d + 1.0;
    if (d < 2.0) return ((-6.0 / 11.0 * (d - 1) + 270.0 / 209.0) * (d - 1) - 156.0 / 209.0) * (d - 1);
    if (d < 3.0) return ((1.0 / 11.0 * (d - 2) - 45.0 / 209.0) * (d - 2) + 26.0 / 209.0) * (d - 2);
    return 0.0;
  }
  const double b = r.b;
  const double c = r.c;
  if (d < 1.0) return ((12 - 9 * b - 6 * c) * d * d * d + (-18 + 12 * b + 6 * c) * d * d + (6 - 2 * b)) / 6.0;
  if (d < 2.0)
    return ((-b - 6 * c) * d * d * d + (6 * b + 30 * c) * d * d + (-12 * b - 48 * c) * d + (8 * b + 24 * c)) /
           6.0;
  return 0.0;
}

// La réduction filtrée telle que le backend TensorRT la définit, en double, bords répétés.
k::RgbImage reduce(const k::RgbImage& in, int ow, int oh, const Reduction& r) {
  const int w = in.width();
  const int h = in.height();
  const double sx = static_cast<double>(w) / ow;
  const double sy = static_cast<double>(h) / oh;
  const double radius = r.kind == ReduceKind::Triangle ? 1.0 : r.kind == ReduceKind::Spline36 ? 3.0 : 2.0;
  k::RgbImage out(ow, oh);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < oh; ++y) {
      for (int x = 0; x < ow; ++x) {
        const double cx = (x + 0.5) * sx - 0.5;
        const double cy = (y + 0.5) * sy - 0.5;
        double sum = 0.0;
        double total = 0.0;
        for (int j = static_cast<int>(std::ceil(cy - radius * sy));
             j <= static_cast<int>(std::floor(cy + radius * sy)); ++j) {
          const double wy = reduce_weight(r, (j - cy) / sy);
          for (int i = static_cast<int>(std::ceil(cx - radius * sx));
               i <= static_cast<int>(std::floor(cx + radius * sx)); ++i) {
            const double wx = reduce_weight(r, (i - cx) / sx);
            sum +=
                wx * wy *
                in.planes[static_cast<std::size_t>(c)].at(std::clamp(i, 0, w - 1), std::clamp(j, 0, h - 1));
            total += wx * wy;
          }
        }
        out.planes[static_cast<std::size_t>(c)].at(x, y) = static_cast<float>(sum / total);
      }
    }
  }
  return out;
}

}  // namespace

TEST_CASE("Réduction fusionnée : tables normalisées, fenêtre dans la source", "[vk_backend][kernels]") {
  for (const Reduction r : {Reduction{.kind = ReduceKind::Triangle}, Reduction{.kind = ReduceKind::Spline36},
                            Reduction{.kind = ReduceKind::Cubic, .b = 0.0F, .c = 0.5F}}) {
    for (const auto [src, dst] :
         {std::array{1920, 1280}, std::array{96, 64}, std::array{10, 10}, std::array{7, 2}}) {
      CAPTURE(static_cast<int>(r.kind), src, dst);
      const auto table = vk::reduction_weights(src, dst, r);
      REQUIRE(table);
      for (int x = 0; x < dst; ++x) {
        const int start = table->start[static_cast<std::size_t>(x)];
        CHECK(start >= 0);
        CHECK(start + static_cast<int>(table->taps) <= src);
        double sum = 0.0;
        for (std::uint32_t j = 0; j < table->taps; ++j)
          sum += table->w[static_cast<std::size_t>(x) * table->taps + j];
        CHECK(std::fabs(sum - 1.0) < 1e-5);
      }
    }
  }
  // À l'échelle 1, le triangle ne garde que le pixel lui-même.
  const auto same = vk::reduction_weights(8, 8, {.kind = ReduceKind::Triangle});
  REQUIRE(same);
  for (int x = 0; x < 8; ++x) {
    for (std::uint32_t j = 0; j < same->taps; ++j) {
      const bool self = same->start[static_cast<std::size_t>(x)] + static_cast<int>(j) == x;
      CHECK(same->w[static_cast<std::size_t>(x) * same->taps + j] == (self ? 1.0F : 0.0F));
    }
  }
  CHECK_FALSE(vk::reduction_weights(64, 96, {.kind = ReduceKind::Triangle}));
  CHECK_FALSE(vk::reduction_weights(96, 64, {.kind = ReduceKind::Cubic, .b = NAN}));
}

TEST_CASE("KernelOps : réductions fusionnées à l'ingestion, contre la définition", "[vk_backend][kernels]") {
  TempDir dir("reduction");
  auto backend = make_backend(dir.path());
  KernelOps& ops = *backend->kernels();
  constexpr int w = 96;
  constexpr int h = 54;
  constexpr int ow = 64;
  constexpr int oh = 36;
  const k::YuvImage image = synthetic_yuv(k::Format::Nv12, w, h, 7);
  const DecodedFrame frame = host_frame(
      image,
      ColorInfo{.matrix = ColorMatrix::Bt709, .range = ColorRange::Limited, .siting = ChromaSiting::Left}, 7);
  auto full = rgb_tensor(*backend, w, h, TensorType::F32);
  auto small = rgb_tensor(*backend, ow, oh, TensorType::F32);
  REQUIRE(ops.ingest(frame, {}, full->tensor()));
  const k::RgbImage base = download(*backend, *full, w, h);

  REQUIRE(ops.ingest(frame, {.reduce = {.kind = ReduceKind::Area}}, small->tensor()));
  const auto area = k::ref::resize(base, ow, oh, k::Filter::Area);
  REQUIRE(area);
  CHECK(max_abs_diff(download(*backend, *small, ow, oh), *area) < 1e-5F);

  for (const Reduction r : {Reduction{.kind = ReduceKind::Triangle}, Reduction{.kind = ReduceKind::Spline36},
                            Reduction{.kind = ReduceKind::Cubic, .b = 0.0F, .c = 0.5F},
                            Reduction{.kind = ReduceKind::Cubic, .b = 1.0F / 3.0F, .c = 1.0F / 3.0F}}) {
    CAPTURE(static_cast<int>(r.kind), r.b, r.c);
    REQUIRE(ops.ingest(frame, {.reduce = r}, small->tensor()));
    CHECK(max_abs_diff(download(*backend, *small, ow, oh), reduce(base, ow, oh, r)) < 1e-5F);
  }
  auto big = rgb_tensor(*backend, 2 * w, h, TensorType::F32);
  const auto refused = ops.ingest(frame, {}, big->tensor());
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::ShapeInvalid);
}
