// GMFSS Fortuna sur le backend Vulkan (gpu/kernel_ops.h) contre kernels/ref : entrée du graphe
// « paire » depuis deux images RGB, softmax splatting par additions atomiques de float.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "backend_rig.h"
#include "kernels/ref/gmfss.h"
#include "kernels/spec/gmfss.h"
#include "kernels/spec/rife.h"
#include "kernels/spec/tolerance.h"

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
