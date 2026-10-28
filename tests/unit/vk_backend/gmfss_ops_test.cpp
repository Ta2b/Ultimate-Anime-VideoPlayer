// GMFSS Fortuna sur le backend Vulkan (gpu/kernel_ops.h) contre kernels/ref : entrée du graphe
// « paire » depuis deux images RGB, softmax splatting par additions atomiques de float.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "backend_rig.h"
#include "kernels/ref/gmfss.h"
#include "kernels/spec/gmfss.h"
#include "kernels/spec/rife.h"
#include "kernels/spec/tolerance.h"
#include "kernels/spec/warp.h"

using namespace ajp;
using namespace ajp::gpu;
using namespace ajp::gpu::vk::test;
namespace k = ajp::kernels;

namespace {

std::vector<TensorType> types_of(const vk::VulkanBackend& backend) {
  std::vector<TensorType> types{TensorType::F32};
  if (backend.profile().target.f16_storage) types.push_back(TensorType::F16);
  return types;
}

std::size_t element_of(TensorType type) {
  return type == TensorType::F16 ? 2 : 4;
}

// count plans synthétiques dans [lo, hi], tels qu'un tenseur de ce type les garde.
std::vector<k::PlaneF> synthetic_planes(int w, int h, int count, std::uint32_t seed, TensorType type,
                                        float lo, float hi) {
  std::vector<k::PlaneF> planes;
  for (int c = 0; c < count; ++c) {
    planes.push_back(
        stored(synthetic_rgb(w, h, seed + static_cast<std::uint32_t>(c), lo, hi), type).planes[0]);
  }
  return planes;
}

float max_diff(const std::vector<k::PlaneF>& a, const std::vector<k::PlaneF>& b) {
  float worst = 0.0F;
  for (std::size_t c = 0; c < a.size(); ++c) {
    for (std::size_t i = 0; i < a[c].data.size(); ++i)
      worst = std::max(worst, std::fabs(a[c].data[i] - b[c].data[i]));
  }
  return worst;
}

}  // namespace

TEST_CASE("KernelOps : entrée de GMFSS depuis deux images RGB", "[vk_backend][kernels][gmfss]") {
  TempDir dir("gmfss_frames");
  auto backend = make_backend(dir.path());
  KernelOps& ops = *backend->kernels();
  constexpr int w = 70;
  constexpr int h = 38;
  for (const TensorType type : types_of(*backend)) {
    CAPTURE(static_cast<int>(type));
    const k::RgbImage a = stored(synthetic_rgb(w, h, 51), type);
    const k::RgbImage b = stored(synthetic_rgb(w, h, 52), type);
    auto ta = rgb_tensor(*backend, w, h, type);
    auto tb = rgb_tensor(*backend, w, h, type);
    write_tensor(*backend, ta->tensor(), rgb_bytes(a, type));
    write_tensor(*backend, tb->tensor(), rgb_bytes(b, type));
    const k::RifeGeometry g = k::rife_geometry(w, h);
    auto frames = rgb_tensor(*backend, g.padded_width, g.padded_height, type, k::kGmfssFramePlanes);
    REQUIRE(ops.gmfss_frames(ta->tensor(), tb->tensor(), frames->tensor()));
    const std::size_t plane =
        static_cast<std::size_t>(g.padded_width) * static_cast<std::size_t>(g.padded_height);
    const auto got = planes_from(read_tensor(*backend, frames->tensor(), element_of(type) * plane * 6),
                                 g.padded_width, g.padded_height, k::kGmfssFramePlanes, type);
    float worst = 0.0F;
    for (int c = 0; c < k::kGmfssFramePlanes; ++c) {
      const k::RgbImage& from = c < 3 ? a : b;
      for (int y = 0; y < g.padded_height; ++y) {
        for (int x = 0; x < g.padded_width; ++x) {
          const int sx = x - g.left;
          const int sy = y - g.top;
          const bool inside = sx >= 0 && sy >= 0 && sx < w && sy < h;
          // La chaîne ingère en plage limitée : le noir de studio y vaut 0 en RGB.
          const float want = inside ? from.planes[static_cast<std::size_t>(c % 3)].at(sx, sy) : 0.0F;
          worst = std::max(worst, std::fabs(got[static_cast<std::size_t>(c)].at(x, y) - want));
        }
      }
    }
    CHECK(worst == 0.0F);
    auto rife_sized = rgb_tensor(*backend, g.padded_width, g.padded_height, type, k::kRifeInputChannels);
    const auto refused = ops.gmfss_frames(ta->tensor(), tb->tensor(), rife_sized->tensor());
    REQUIRE_FALSE(refused);
    CHECK(refused.error().code == RefusalCode::ShapeInvalid);
  }
}

TEST_CASE("KernelOps : un tenseur rempli d'une valeur, l'instant de GMFSS Union",
          "[vk_backend][kernels][gmfss]") {
  TempDir dir("tensor_fill");
  auto backend = make_backend(dir.path());
  KernelOps& ops = *backend->kernels();
  for (const TensorType type : types_of(*backend)) {
    CAPTURE(static_cast<int>(type));
    for (const auto& [w, h, c] : {std::array{1, 1, 1}, std::array{37, 5, 3}}) {
      auto t = rgb_tensor(*backend, w, h, type, c);
      REQUIRE(ops.fill(t->tensor(), 0.375F));
      const std::size_t count = static_cast<std::size_t>(w * h * c);
      const auto got =
          planes_from(read_tensor(*backend, t->tensor(), element_of(type) * count), w, h, c, type);
      for (const k::PlaneF& plane : got) {
        for (const float v : plane.data) CHECK(v == 0.375F);
      }
    }
  }
}

TEST_CASE("KernelOps : softmax splatting contre la référence", "[vk_backend][kernels][gmfss]") {
  TempDir dir("soft_splat");
  auto backend = make_backend(dir.path());
  KernelOps& ops = *backend->kernels();
  constexpr int w = 40;
  constexpr int h = 24;
  // 19 plans : deux groupes de l'axe z, le second incomplet.
  constexpr int channels = 19;
  for (const TensorType type : types_of(*backend)) {
    CAPTURE(static_cast<int>(type));
    const std::vector<k::PlaneF> source = synthetic_planes(w, h, channels, 61, type, 0.0F, 1.0F);
    const std::vector<k::PlaneF> flow = synthetic_planes(w, h, 2, 81, type, -6.0F, 6.0F);
    const std::vector<k::PlaneF> metric =
        synthetic_planes(w, h, 1, 91, type, -k::kGmfssMetricLimit, k::kGmfssMetricLimit);
    const k::ref::SplatMotion motion{{flow[0], flow[1], metric[0]}};
    auto ts = rgb_tensor(*backend, w, h, type, channels);
    auto tm = rgb_tensor(*backend, w, h, type, k::kGmfssMotionChannels);
    auto out = rgb_tensor(*backend, w, h, type, channels);
    write_tensor(*backend, ts->tensor(), planes_bytes(source, type));
    write_tensor(*backend, tm->tensor(), planes_bytes(motion.planes, type));
    for (const float s : {0.5F, 0.25F, 1.0F}) {
      CAPTURE(s);
      const auto done = ops.soft_splat(ts->tensor(), tm->tensor(), s, out->tensor());
      if (!done && done.error().code == RefusalCode::FeatureUnavailable) SKIP(done.error().detail);
      REQUIRE(done);
      const auto want = k::ref::soft_splat(source, motion, s);
      REQUIRE(want);
      const std::size_t bytes = element_of(type) * static_cast<std::size_t>(w * h * channels);
      const auto got = planes_from(read_tensor(*backend, out->tensor(), bytes), w, h, channels, type);
      // Seul l'ordre des additions atomiques change les arrondis, puis le float16 de la sortie.
      CHECK(max_diff(got, *want) <= (type == TensorType::F16 ? k::tolerance::kRgbF16 : 1e-5F));
    }
    const auto late = ops.soft_splat(ts->tensor(), tm->tensor(), 1.5F, out->tensor());
    REQUIRE_FALSE(late);
    CHECK(late.error().code == RefusalCode::FieldOutOfRange);
    const auto shape = ops.soft_splat(ts->tensor(), tm->tensor(), 0.5F, tm->tensor());
    REQUIRE_FALSE(shape);
    CHECK(shape.error().code == RefusalCode::ShapeInvalid);
  }
}

TEST_CASE("KernelOps : les six splats d'un instant en un passage, chacun contre la référence",
          "[vk_backend][kernels][gmfss]") {
  // Trois niveaux de tailles et de plans différents, deux côtés aux instants t et 1 − t : les
  // sommes de chacun vivent côte à côte dans un même tampon, et aucun ne déborde sur l'autre.
  TempDir dir("soft_splat_all");
  auto backend = make_backend(dir.path());
  KernelOps& ops = *backend->kernels();
  constexpr float t = 0.3F;
  struct Level {
    int w;
    int h;
    int channels;
  };
  constexpr std::array levels{Level{40, 24, 19}, Level{20, 12, 7}, Level{10, 6, 3}};
  for (const TensorType type : types_of(*backend)) {
    CAPTURE(static_cast<int>(type));
    std::vector<std::vector<k::PlaneF>> sources;
    std::vector<k::ref::SplatMotion> motions;
    std::vector<std::unique_ptr<TensorBuffer>> tensors;
    std::vector<KernelOps::SplatJob> jobs;
    std::uint32_t seed = 7;
    for (int side = 0; side < 2; ++side) {
      for (const Level& level : levels) {
        sources.push_back(synthetic_planes(level.w, level.h, level.channels, seed++, type, 0.0F, 1.0F));
        const std::vector<k::PlaneF> flow = synthetic_planes(level.w, level.h, 2, seed++, type, -4.0F, 4.0F);
        const std::vector<k::PlaneF> metric =
            synthetic_planes(level.w, level.h, 1, seed++, type, -k::kGmfssMetricLimit, k::kGmfssMetricLimit);
        motions.push_back({{flow[0], flow[1], metric[0]}});
        tensors.push_back(rgb_tensor(*backend, level.w, level.h, type, level.channels));
        tensors.push_back(rgb_tensor(*backend, level.w, level.h, type, k::kGmfssMotionChannels));
        tensors.push_back(rgb_tensor(*backend, level.w, level.h, type, level.channels));
        const std::size_t n = tensors.size();
        write_tensor(*backend, tensors[n - 3]->tensor(), planes_bytes(sources.back(), type));
        write_tensor(*backend, tensors[n - 2]->tensor(), planes_bytes(motions.back().planes, type));
        jobs.push_back({.source = tensors[n - 3]->tensor(),
                        .motion = tensors[n - 2]->tensor(),
                        .s = k::gmfss_side_time(side, t),
                        .out = tensors[n - 1]->tensor()});
      }
    }
    const auto done = ops.soft_splat_all(jobs);
    if (!done && done.error().code == RefusalCode::FeatureUnavailable) SKIP(done.error().detail);
    REQUIRE(done);
    for (std::size_t j = 0; j < jobs.size(); ++j) {
      CAPTURE(j);
      const Level& level = levels[j % levels.size()];
      const auto want = k::ref::soft_splat(sources[j], motions[j], jobs[j].s);
      REQUIRE(want);
      const std::size_t bytes =
          element_of(type) * static_cast<std::size_t>(level.w * level.h * level.channels);
      const auto got =
          planes_from(read_tensor(*backend, jobs[j].out, bytes), level.w, level.h, level.channels, type);
      CHECK(max_diff(got, *want) <= (type == TensorType::F16 ? k::tolerance::kRgbF16 : 1e-5F));
    }
    // Un travail faux refuse le lot entier avant toute soumission.
    jobs[1].out = jobs[1].motion;
    const auto shape = ops.soft_splat_all(jobs);
    REQUIRE_FALSE(shape);
    CHECK(shape.error().code == RefusalCode::ShapeInvalid);
  }
}

TEST_CASE("KernelOps : le flot du niveau 1 pousse l'image de travail et l'agrandie, puis la composition",
          "[vk_backend][kernels][gmfss]") {
  // Travail 20 × 12 dans des plans remplis 32 × 16, fenêtre en (6, 2) ; flot du niveau 1 en 16 × 8 ;
  // sortie 40 × 24. Les quatre splats passent ensemble, avec leur couverture ; la composition se
  // compare à la référence nourrie de ce que le GPU a splatté.
  TempDir dir("gmfss_compose");
  auto backend = make_backend(dir.path());
  KernelOps& ops = *backend->kernels();
  constexpr int w = 20;
  constexpr int h = 12;
  constexpr int mw = 16;
  constexpr int mh = 8;
  constexpr int ow = 40;
  constexpr int oh = 24;
  constexpr int fw = 32;
  constexpr int fh = 16;
  constexpr k::Window window{.x = 6, .y = 2, .width = w, .height = h};
  constexpr k::MotionGrid work_grid{.scale = 2.0F, .x = 3.0F, .y = 1.0F};
  constexpr k::MotionGrid spanned_grid{.scale = 4.0F, .x = 3.0F, .y = 1.0F};
  constexpr float t = 0.3F;
  for (const TensorType type : types_of(*backend)) {
    CAPTURE(static_cast<int>(type));
    std::uint32_t seed = 101;
    const auto make = [&](int width, int height, int planes, float lo, float hi) {
      auto tensor = rgb_tensor(*backend, width, height, type, planes);
      const std::vector<k::PlaneF> values = synthetic_planes(width, height, planes, seed++, type, lo, hi);
      write_tensor(*backend, tensor->tensor(), planes_bytes(values, type));
      return std::pair{std::move(tensor), values};
    };
    auto [ta, work_a] = make(w, h, 3, 0.0F, 1.0F);
    auto [tb, work_b] = make(w, h, 3, 0.0F, 1.0F);
    auto [sa, span_a] = make(ow, oh, 3, 0.0F, 1.0F);
    auto [sb, span_b] = make(ow, oh, 3, 0.0F, 1.0F);
    auto [tf, fusion] = make(fw, fh, 3, 0.0F, 1.0F);
    std::array<k::ref::SplatMotion, 2> motions;
    std::array<std::unique_ptr<TensorBuffer>, 2> tm;
    for (std::size_t side = 0; side < 2; ++side) {
      const std::vector<k::PlaneF> flow = synthetic_planes(mw, mh, 2, seed++, type, -1.5F, 1.5F);
      const std::vector<k::PlaneF> metric =
          synthetic_planes(mw, mh, 1, seed++, type, -k::kGmfssMetricLimit, k::kGmfssMetricLimit);
      motions[side] = {{flow[0], flow[1], metric[0]}};
      tm[side] = rgb_tensor(*backend, mw, mh, type, k::kGmfssMotionChannels);
      write_tensor(*backend, tm[side]->tensor(), planes_bytes(motions[side].planes, type));
    }
    std::array<std::unique_ptr<TensorBuffer>, 2> low{rgb_tensor(*backend, w, h, type, 4),
                                                     rgb_tensor(*backend, w, h, type, 4)};
    std::array<std::unique_ptr<TensorBuffer>, 2> high{rgb_tensor(*backend, ow, oh, type, 4),
                                                      rgb_tensor(*backend, ow, oh, type, 4)};
    const std::array<const std::vector<k::PlaneF>*, 2> works{&work_a, &work_b};
    const std::array<const std::vector<k::PlaneF>*, 2> spans{&span_a, &span_b};
    const std::array<const Tensor*, 2> work_tensors{&ta->tensor(), &tb->tensor()};
    const std::array<const Tensor*, 2> span_tensors{&sa->tensor(), &sb->tensor()};
    std::vector<KernelOps::SplatJob> jobs;
    for (std::size_t side = 0; side < 2; ++side) {
      const float s = k::gmfss_side_time(static_cast<int>(side), t);
      jobs.push_back({.source = *work_tensors[side],
                      .motion = tm[side]->tensor(),
                      .s = s,
                      .out = low[side]->tensor(),
                      .grid = work_grid});
      jobs.push_back({.source = *span_tensors[side],
                      .motion = tm[side]->tensor(),
                      .s = s,
                      .out = high[side]->tensor(),
                      .grid = spanned_grid});
    }
    const auto done = ops.soft_splat_all(jobs);
    if (!done && done.error().code == RefusalCode::FeatureUnavailable) SKIP(done.error().detail);
    REQUIRE(done);
    const float tolerance = type == TensorType::F16 ? k::tolerance::kRgbF16 : 1e-5F;
    std::array<k::ref::Splatted, 2> got_low;
    std::array<k::ref::Splatted, 2> got_high;
    const auto splatted = [&](const Tensor& tensor, int width, int height) {
      const std::size_t bytes = element_of(type) * static_cast<std::size_t>(width * height * 4);
      std::vector<k::PlaneF> planes =
          planes_from(read_tensor(*backend, tensor, bytes), width, height, 4, type);
      k::ref::Splatted out{.coverage = planes[3]};
      planes.pop_back();
      out.planes = std::move(planes);
      return out;
    };
    for (std::size_t side = 0; side < 2; ++side) {
      CAPTURE(side);
      const float s = k::gmfss_side_time(static_cast<int>(side), t);
      got_low[side] = splatted(low[side]->tensor(), w, h);
      got_high[side] = splatted(high[side]->tensor(), ow, oh);
      const auto want_low = k::ref::soft_splat_at(*works[side], motions[side], s, work_grid);
      const auto want_high = k::ref::soft_splat_at(*spans[side], motions[side], s, spanned_grid);
      REQUIRE((want_low && want_high));
      CHECK(max_diff(got_low[side].planes, want_low->planes) <= tolerance);
      CHECK(max_diff({got_low[side].coverage}, {want_low->coverage}) <= tolerance);
      CHECK(max_diff(got_high[side].planes, want_high->planes) <= tolerance);
      CHECK(max_diff({got_high[side].coverage}, {want_high->coverage}) <= tolerance);
    }
    auto out = rgb_tensor(*backend, ow, oh, type);
    REQUIRE(ops.gmfss_compose(high[0]->tensor(), high[1]->tensor(), low[0]->tensor(), low[1]->tensor(),
                              tf->tensor(), window, t, out->tensor()));
    const auto want =
        k::ref::gmfss_compose(got_high[0], got_high[1], got_low[0], got_low[1], fusion, window, t);
    REQUIRE(want);
    const std::size_t bytes = element_of(type) * static_cast<std::size_t>(ow * oh * 3);
    const auto composed = planes_from(read_tensor(*backend, out->tensor(), bytes), ow, oh, 3, type);
    CHECK(max_diff(composed, *want) <= tolerance);
    // Une fenêtre hors de la fusion, ou une sortie qui n'est pas un multiple du travail : refus.
    const auto outside = ops.gmfss_compose(
        high[0]->tensor(), high[1]->tensor(), low[0]->tensor(), low[1]->tensor(), tf->tensor(),
        k::Window{.x = 14, .y = 2, .width = w, .height = h}, t, out->tensor());
    REQUIRE_FALSE(outside);
    CHECK(outside.error().code == RefusalCode::ShapeInvalid);
    const auto uneven = ops.gmfss_compose(high[0]->tensor(), high[1]->tensor(), high[0]->tensor(),
                                          high[1]->tensor(), tf->tensor(), window, t, out->tensor());
    REQUIRE_FALSE(uneven);
    CHECK(uneven.error().code == RefusalCode::ShapeInvalid);
  }
}
