// Extensions de la tranche verticale (gpu/kernel_ops.h) sur le backend Vulkan, contre les mêmes
// références que le backend TensorRT : réduction fusionnée à l'ingestion, entrée de RIFE depuis
// le RGB, mesures de la paire, plus grand déplacement, warp à poses gardées.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "backend_rig.h"
#include "gpu/vulkan/backend/reduce_weights.h"
#include "kernels/ref/resample.h"
#include "kernels/ref/rife.h"
#include "kernels/ref/warp.h"
#include "kernels/spec/rife.h"
#include "kernels/spec/tolerance.h"

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

std::vector<TensorType> types_of(const vk::VulkanBackend& backend) {
  std::vector<TensorType> types{TensorType::F32};
  if (backend.profile().target.f16_storage) types.push_back(TensorType::F16);
  return types;
}

// Le mouvement de référence et ses tenseurs : plans 32 × 20, fenêtre 24 × 16 en (4, 2).
struct MotionPair {
  k::ref::Motion ref;
  std::unique_ptr<TensorBuffer> flow;
  std::unique_ptr<TensorBuffer> mask;
  Motion gpu() const { return {.flow = flow->tensor(), .mask = mask->tensor(), .area = ref.area}; }
};

MotionPair motion_of(vk::VulkanBackend& backend, TensorType type, std::uint32_t seed, float scale) {
  MotionPair m;
  m.ref.area = {4, 2, 24, 16};
  for (std::size_t c = 0; c < 4; ++c) {
    m.ref.flow[c] =
        stored(synthetic_rgb(32, 20, seed + static_cast<std::uint32_t>(c), -scale, scale), type).planes[0];
  }
  m.ref.mask = stored(synthetic_rgb(32, 20, seed + 4), type).planes[0];
  m.flow = rgb_tensor(backend, 32, 20, type, 4);
  m.mask = rgb_tensor(backend, 32, 20, type, 1);
  write_tensor(backend, m.flow->tensor(), planes_bytes(m.ref.flow, type));
  write_tensor(backend, m.mask->tensor(), planes_bytes(std::span(&m.ref.mask, 1), type));
  return m;
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

TEST_CASE("KernelOps : entrée de RIFE depuis deux images RGB, puis la paire suivante, contre kernels/ref",
          "[vk_backend][kernels]") {
  TempDir dir("rife_input");
  auto backend = make_backend(dir.path());
  KernelOps& ops = *backend->kernels();
  for (const TensorType type : types_of(*backend)) {
    CAPTURE(static_cast<int>(type));
    const k::RgbImage a = stored(synthetic_rgb(70, 38, 21), type);
    const k::RgbImage b = stored(synthetic_rgb(70, 38, 22), type);
    auto ta = rgb_tensor(*backend, 70, 38, type);
    auto tb = rgb_tensor(*backend, 70, 38, type);
    write_tensor(*backend, ta->tensor(), rgb_bytes(a, type));
    write_tensor(*backend, tb->tensor(), rgb_bytes(b, type));
    const k::RifeGeometry g = k::rife_geometry(70, 38);
    // ref::rife_input prend les images déjà remplies : noir de studio de la chaîne (0) autour.
    const auto pad = [&](const k::RgbImage& i) {
      k::RgbImage p(g.padded_width, g.padded_height, 0.0F);
      for (std::size_t c = 0; c < 3; ++c) {
        for (int y = 0; y < 38; ++y) {
          for (int x = 0; x < 70; ++x) p.planes[c].at(x + g.left, y + g.top) = i.planes[c].at(x, y);
        }
      }
      return p;
    };
    // v1 : les 11 plans ; v2 : les 7 premiers, la grille est dans le graphe.
    for (const int planes : {k::kRifeInputChannels, k::kRifeV2InputChannels}) {
      CAPTURE(planes);
      auto input = rgb_tensor(*backend, g.padded_width, g.padded_height, type, planes);
      const auto check = [&](const k::RgbImage& first, const k::RgbImage& second, float t) {
        const auto want = k::ref::rife_input(pad(first), pad(second), t);
        REQUIRE(want);
        const std::size_t element = type == TensorType::F16 ? 2 : 4;
        const std::size_t plane =
            static_cast<std::size_t>(g.padded_width) * static_cast<std::size_t>(g.padded_height);
        const auto got = planes_from(
            read_tensor(*backend, input->tensor(), element * plane * static_cast<std::size_t>(planes)),
            g.padded_width, g.padded_height, planes, type);
        for (int c = 0; c < planes; ++c) {
          CAPTURE(c);
          k::RgbImage expect(g.padded_width, g.padded_height);
          expect.planes[0] = (*want)[static_cast<std::size_t>(c)];
          float worst = 0.0F;
          const k::PlaneF stored_plane = stored(expect, type).planes[0];
          for (std::size_t i = 0; i < plane; ++i)
            worst =
                std::max(worst, std::fabs(got[static_cast<std::size_t>(c)].data[i] - stored_plane.data[i]));
          CHECK(worst <= k::tolerance::kRgbF16);
        }
      };
      for (const float t : {0.5F, 0.25F}) {
        CAPTURE(t);
        REQUIRE(ops.rife_input(ta->tensor(), tb->tensor(), t, input->tensor()));
        check(a, b, t);
      }
      // Paire suivante : les images et l'instant seuls, le maillage de rife_input reste.
      REQUIRE(ops.rife_pair(tb->tensor(), ta->tensor(), 0.75F, input->tensor()));
      check(b, a, 0.75F);
    }
    auto input = rgb_tensor(*backend, g.padded_width, g.padded_height, type, k::kRifeInputChannels);
    auto eight = rgb_tensor(*backend, g.padded_width, g.padded_height, type, 8);
    for (const bool pair : {false, true}) {
      CAPTURE(pair);
      const auto fill = [&](float t, const Tensor& into) {
        return pair ? ops.rife_pair(ta->tensor(), tb->tensor(), t, into)
                    : ops.rife_input(ta->tensor(), tb->tensor(), t, into);
      };
      const auto odd = fill(0.5F, eight->tensor());
      REQUIRE_FALSE(odd);
      CHECK(odd.error().code == RefusalCode::ShapeInvalid);
      const auto refused = fill(1.0F, input->tensor());
      REQUIRE_FALSE(refused);
      CHECK(refused.error().code == RefusalCode::FieldOutOfRange);
    }
  }
}

TEST_CASE("KernelOps : mesures d'une paire RGB rendues tout de suite", "[vk_backend][kernels]") {
  TempDir dir("scores_now");
  auto backend = make_backend(dir.path());
  KernelOps& ops = *backend->kernels();
  constexpr int w = 70;
  constexpr int h = 38;
  const k::RgbImage a = synthetic_rgb(w, h, 31);
  k::RgbImage b = a;
  for (int y = 32; y < h; ++y) {
    for (int x = 0; x < 32; ++x) b.planes[1].at(x, y) = std::min(1.0F, b.planes[1].at(x, y) + 0.25F);
  }
  auto ta = rgb_tensor(*backend, w, h, TensorType::F32);
  auto tb = rgb_tensor(*backend, w, h, TensorType::F32);
  write_tensor(*backend, ta->tensor(), rgb_bytes(a, TensorType::F32));
  write_tensor(*backend, tb->tensor(), rgb_bytes(b, TensorType::F32));

  const auto same = ops.scores_now(ta->tensor(), ta->tensor());
  REQUIRE(same);
  CHECK(same->identical);
  CHECK(same->scene_score == 0.0);
  CHECK(same->worst_block == 0);

  const auto got = ops.scores_now(ta->tensor(), tb->tensor());
  REQUIRE(got);
  // La définition de rife_kernels.cu, en double : luma BT.709 du RGB, blocs 32 × 32 tronqués.
  double total = 0.0;
  std::uint32_t worst = 0;
  for (int by = 0; by < h; by += 32) {
    for (int bx = 0; bx < w; bx += 32) {
      double block = 0.0;
      const int x1 = std::min(bx + 32, w);
      const int y1 = std::min(by + 32, h);
      for (int y = by; y < y1; ++y) {
        for (int x = bx; x < x1; ++x) {
          const auto luma = [&](const k::RgbImage& i) {
            return 0.2126 * i.planes[0].at(x, y) + 0.7152 * i.planes[1].at(x, y) +
                   0.0722 * i.planes[2].at(x, y);
          };
          block += std::fabs(luma(a) - luma(b));
        }
      }
      total += block;
      worst = std::max(worst, static_cast<std::uint32_t>(block / ((x1 - bx) * (y1 - by)) * 219.0 * 256.0));
    }
  }
  const k::RifeGeometry g = k::rife_geometry(w, h);
  const double scene = total * (219.0 / 255.0) / (g.padded_width * g.padded_height);
  CHECK_FALSE(got->identical);
  CHECK(std::fabs(got->scene_score - scene) <= 1e-5 * scene);
  CHECK(std::abs(static_cast<int>(got->worst_block) - static_cast<int>(worst)) <= 1);
  CHECK(got->worst_block > 0);

  // En deux temps : deux mesures en vol, relues dans n'importe quel ordre, chacune une fois.
  const auto first = ops.scores_start(ta->tensor(), tb->tensor());
  const auto second = ops.scores_start(ta->tensor(), ta->tensor());
  REQUIRE(first);
  REQUIRE(second);
  const auto later = ops.scores_take(*second);
  REQUIRE(later);
  CHECK(later->identical);
  const auto earlier = ops.scores_take(*first);
  REQUIRE(earlier);
  CHECK(earlier->scene_score == got->scene_score);
  CHECK(earlier->worst_block == got->worst_block);
  CHECK_FALSE(ops.scores_take(*first));
  // Un ticket dépassé par kPendingReadbacks lancements est périmé.
  const auto old = ops.scores_start(ta->tensor(), tb->tensor());
  REQUIRE(old);
  for (std::size_t k = 0; k < kPendingReadbacks; ++k) REQUIRE(ops.scores_start(ta->tensor(), tb->tensor()));
  CHECK_FALSE(ops.scores_take(*old));
}

TEST_CASE("KernelOps : plus grand déplacement rendu tout de suite, contre kernels/ref",
          "[vk_backend][kernels]") {
  TempDir dir("flow_max");
  auto backend = make_backend(dir.path());
  KernelOps& ops = *backend->kernels();
  for (const TensorType type : types_of(*backend)) {
    CAPTURE(static_cast<int>(type));
    const MotionPair m = motion_of(*backend, type, 41, 7.0F);
    const auto got = ops.flow_max_now(m.gpu());
    REQUIRE(got);
    const auto want = k::ref::flow_max(m.ref);
    REQUIRE(want);
    CHECK(*got == *want);
  }
}

TEST_CASE("KernelOps : warp à poses gardées, contre kernels/ref", "[vk_backend][kernels]") {
  TempDir dir("poses");
  auto backend = make_backend(dir.path());
  KernelOps& ops = *backend->kernels();
  for (const TensorType type : types_of(*backend)) {
    CAPTURE(static_cast<int>(type));
    const k::RgbImage a = stored(synthetic_rgb(24, 16, 51), type);
    const k::RgbImage b = stored(synthetic_rgb(24, 16, 52), type);
    auto ta = rgb_tensor(*backend, 24, 16, type);
    auto tb = rgb_tensor(*backend, 24, 16, type);
    auto out = rgb_tensor(*backend, 24, 16, type);
    write_tensor(*backend, ta->tensor(), rgb_bytes(a, type));
    write_tensor(*backend, tb->tensor(), rgb_bytes(b, type));
    const MotionPair now = motion_of(*backend, type, 53, 3.0F);
    const MotionPair before = motion_of(*backend, type, 58, 1.0F);
    const Motion previous = before.gpu();
    for (const bool held : {false, true}) {
      for (const float t : {0.25F, 0.5F, 0.75F}) {
        CAPTURE(held, t);
        REQUIRE(ops.warp_blend_poses(ta->tensor(), tb->tensor(), now.gpu(), held ? nullptr : &previous, t,
                                     out->tensor()));
        const auto want = k::ref::warp_blend_poses(a, b, now.ref, held ? nullptr : &before.ref, t);
        REQUIRE(want);
        CHECK(max_abs_diff(download(*backend, *out, 24, 16), stored(*want, type)) <= k::tolerance::kWarp);
      }
    }
    // Un mouvement précédent d'une autre fenêtre est refusé.
    Motion shifted = previous;
    shifted.area.x = 0;
    const auto refused =
        ops.warp_blend_poses(ta->tensor(), tb->tensor(), now.gpu(), &shifted, 0.25F, out->tensor());
    REQUIRE_FALSE(refused);
    CHECK(refused.error().code == RefusalCode::ShapeInvalid);
  }
}
