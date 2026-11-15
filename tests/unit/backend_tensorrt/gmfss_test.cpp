// GMFSS Fortuna Union par le backend TensorRT (gpu/kernel_ops.h) contre kernels/ref : l'entrée de la paire et
// le softmax splatting par additions atomiques CUDA, sur sa grille avec couverture, et la composition à la
// taille de travail. Sans GPU NVIDIA, le test se SKIP.
#include "kernels/ref/gmfss.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "backend_fixture.h"
#include "kernels/ref/half.h"
#include "kernels/spec/gmfss.h"
#include "kernels/spec/rife.h"
#include "kernels/spec/tolerance.h"

namespace {

using namespace ajp;
using namespace ajp::test;
namespace k = ajp::kernels;

constexpr int kW = 40;
constexpr int kH = 24;

// Des plans dans [lo, hi], déjà arrondis en FP16 comme le tenseur les garde.
std::vector<k::PlaneF> planes(int count, int seed, float lo, float hi, int w = kW, int h = kH) {
  std::vector<k::PlaneF> out;
  for (int c = 0; c < count; ++c) {
    k::PlaneF plane(w, h);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const float u = 0.5F + 0.5F * std::sin(0.37F * static_cast<float>(x + seed * 7 + c * 3) +
                                               0.23F * static_cast<float>(y - seed));
        plane.at(x, y) = k::ref::half_to_float(k::ref::float_to_half(lo + (hi - lo) * u));
      }
    }
    out.push_back(std::move(plane));
  }
  return out;
}

std::vector<float> flat(const std::vector<k::PlaneF>& planes) {
  std::vector<float> out;
  for (const k::PlaneF& plane : planes) out.insert(out.end(), plane.data.begin(), plane.data.end());
  return out;
}

float worst_of(const std::vector<float>& got, const std::vector<float>& expected) {
  REQUIRE(got.size() == expected.size());
  float worst = 0.0F;
  for (std::size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::fabs(got[i] - expected[i]));
  return worst;
}

}  // namespace

TEST_CASE("Backend TensorRT : les splats d'un instant en un passage, l'instant rempli",
          "[backend_tensorrt][cuda][gmfss]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  REQUIRE(loaded.backend->capabilities().gmfss);
  gpu::KernelOps& ops = *loaded.backend->kernels();
  auto alloc = [&](int channels) {
    auto t = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, channels, kH, kW}});
    REQUIRE(t);
    return std::move(*t);
  };
  // ±6 px : les contributions restent dans la fenêtre partagée de leur tuile. De −14 à 22 px : la fenêtre
  // suit le déplacement moyen de la tuile, les écarts au-delà de sa marge vont directement aux sommes
  // globales, et une partie sort de l'image.
  for (const auto& [lo, hi] : {std::pair{-6.0F, 6.0F}, std::pair{-14.0F, 22.0F}}) {
    CAPTURE(lo, hi);
    constexpr float t = 0.3F;
    // Deux côtés, trois sources de plans différents (19 : deux groupes, le second incomplet), la dernière
    // écrite dans une tranche d'un tenseur plus grand (l'image du niveau 1 dans warped{k}_1).
    constexpr std::array channels{19, 7, 3};
    std::vector<std::vector<k::PlaneF>> sources;
    std::vector<k::ref::SplatMotion> motions;
    std::vector<std::unique_ptr<gpu::TensorBuffer>> tensors;
    std::vector<gpu::KernelOps::SplatJob> jobs;
    int seed = 10;
    for (int side = 0; side < 2; ++side) {
      for (const int c : channels) {
        sources.push_back(planes(c, seed++, 0.0F, 1.0F));
        const std::vector<k::PlaneF> flow = planes(2, seed++, lo, hi);
        const std::vector<k::PlaneF> metric = planes(1, seed++, -k::kGmfssMetricLimit, k::kGmfssMetricLimit);
        motions.push_back({{flow[0], flow[1], metric[0]}});
        tensors.push_back(alloc(c));
        tensors.push_back(alloc(k::kGmfssMotionChannels));
        tensors.push_back(alloc(c + 2));
        const std::size_t n = tensors.size();
        upload(tensors[n - 3]->tensor(), flat(sources.back()));
        upload(tensors[n - 2]->tensor(),
               flat(std::vector<k::PlaneF>(motions.back().planes.begin(), motions.back().planes.end())));
        gpu::Tensor out = tensors[n - 1]->tensor();  // plans 2 à c + 1
        out.shape.dims[1] = c;
        out.offset += 2U * kW * kH * sizeof(std::uint16_t);
        jobs.push_back({.source = tensors[n - 3]->tensor(),
                        .motion = tensors[n - 2]->tensor(),
                        .s = k::gmfss_side_time(side, t),
                        .out = out});
      }
    }
    auto time = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 1, 1, 1}});
    REQUIRE(time);
    REQUIRE(ops.soft_splat_fill(jobs, (*time)->tensor(), t));
    for (const float v : download((*time)->tensor()))
      CHECK(v == k::ref::half_to_float(k::ref::float_to_half(t)));
    for (std::size_t j = 0; j < jobs.size(); ++j) {
      CAPTURE(j);
      const auto want = k::ref::soft_splat(sources[j], motions[j], jobs[j].s);
      REQUIRE(want);
      const std::vector<float> whole = download(tensors[3 * j + 2]->tensor());
      const std::size_t plane = static_cast<std::size_t>(kW) * kH;
      const std::vector<float> got(whole.begin() + static_cast<std::ptrdiff_t>(2 * plane), whole.end());
      // Seul l'ordre des additions change les arrondis (atomiques, fenêtre partagée puis globale), puis le
      // FP16 de la sortie.
      CHECK(worst_of(got, flat(*want)) <= k::tolerance::kRgbF16);
    }
    if (lo < 0.0F && hi > 10.0F) {
      // L'instant dans la mémoire d'un splat, trop de splats, une sortie qui est une entrée, un instant hors
      // de [0, 1] : le lot entier est refusé.
      const auto shared = ops.soft_splat_fill(jobs, jobs[1].source, t);
      REQUIRE_FALSE(shared);
      CHECK(shared.error().code == RefusalCode::ShapeInvalid);
      std::vector<gpu::KernelOps::SplatJob> many(9, jobs[0]);
      const auto too_many = ops.soft_splat_fill(many, (*time)->tensor(), t);
      REQUIRE_FALSE(too_many);
      CHECK(too_many.error().code == RefusalCode::FieldOutOfRange);
      const auto late = ops.soft_splat_fill(jobs, (*time)->tensor(), 1.5F);
      REQUIRE_FALSE(late);
      CHECK(late.error().code == RefusalCode::FieldOutOfRange);
      jobs[2].out = jobs[2].motion;
      const auto shape = ops.soft_splat_fill(jobs, (*time)->tensor(), t);
      REQUIRE_FALSE(shape);
      CHECK(shape.error().code == RefusalCode::ShapeInvalid);
    }
  }
}

TEST_CASE("Backend TensorRT : l'entrée de la paire de GMFSS, a puis b dans leur fenêtre, bords noirs",
          "[backend_tensorrt][cuda][gmfss]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  gpu::KernelOps& ops = *loaded.backend->kernels();
  constexpr int w = 50;
  constexpr int h = 30;
  const k::RifeGeometry g = k::rife_geometry(w, h, k::kGmfssAlign);
  auto a = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, h, w}});
  auto b = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, h, w}});
  auto frames = loaded.backend->alloc_tensor(
      gpu::TensorType::F16, {.dims = {1, k::kGmfssFramePlanes, g.padded_height, g.padded_width}});
  REQUIRE((a && b && frames));
  const std::vector<k::PlaneF> pa = planes(3, 1, 0.0F, 1.0F, w, h);
  const std::vector<k::PlaneF> pb = planes(3, 2, 0.0F, 1.0F, w, h);
  upload((*a)->tensor(), flat(pa));
  upload((*b)->tensor(), flat(pb));
  REQUIRE(ops.gmfss_frames((*a)->tensor(), (*b)->tensor(), (*frames)->tensor()));
  const std::vector<float> got = download((*frames)->tensor());
  const std::size_t plane =
      static_cast<std::size_t>(g.padded_width) * static_cast<std::size_t>(g.padded_height);
  for (int c = 0; c < k::kGmfssFramePlanes; ++c) {
    const k::PlaneF& source = c < 3 ? pa[static_cast<std::size_t>(c)] : pb[static_cast<std::size_t>(c - 3)];
    for (int y = 0; y < g.padded_height; ++y) {
      for (int x = 0; x < g.padded_width; ++x) {
        const bool inside = x >= g.left && x < g.left + w && y >= g.top && y < g.top + h;
        const float want = inside ? source.at(x - g.left, y - g.top) : 0.0F;
        const float value = got[static_cast<std::size_t>(c) * plane +
                                static_cast<std::size_t>(y) * static_cast<std::size_t>(g.padded_width) +
                                static_cast<std::size_t>(x)];
        if (value != want)
          FAIL_CHECK("plan " << c << " (" << x << ", " << y << ") : " << value << " au lieu de " << want);
      }
    }
  }
  const auto odd = ops.gmfss_frames((*a)->tensor(), (*frames)->tensor(), (*frames)->tensor());
  REQUIRE_FALSE(odd);
  CHECK(odd.error().code == RefusalCode::ShapeInvalid);
}

TEST_CASE("Backend TensorRT : GMFSS, splat sur la grille du mouvement et composition à la taille de travail",
          "[backend_tensorrt][cuda][gmfss]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  gpu::KernelOps& ops = *loaded.backend->kernels();
  auto alloc = [&](int channels, int w, int h) {
    auto t = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, channels, h, w}});
    REQUIRE(t);
    return std::move(*t);
  };
  // Le mouvement sur 24 × 16 (le niveau 1 de plans remplis 48 × 32), l'image de RIFE 40 × 24 dans la
  // fenêtre (4, 4), celle de travail 60 × 36 (×1,5).
  constexpr int kMw = 24;
  constexpr int kMh = 16;
  constexpr int kRw = 40;
  constexpr int kRh = 24;
  constexpr int kOw = 60;
  constexpr int kOh = 36;
  const k::Window window{.x = 4, .y = 4, .width = kRw, .height = kRh};
  const k::MotionGrid low_grid{.scale = 2.0F, .x = 2.0F, .y = 2.0F};
  const k::MotionGrid high_grid{.scale = 2.0F * kOw / kRw, .x = 2.0F, .y = 2.0F};
  std::vector<std::unique_ptr<gpu::TensorBuffer>> keep;
  std::vector<gpu::KernelOps::SplatJob> jobs;
  std::array<k::ref::Splatted, 4> want;  // haut a, haut b, bas a, bas b
  for (int side = 0; side < 2; ++side) {
    const std::vector<k::PlaneF> flow = planes(2, 40 + side, -3.0F, 3.0F, kMw, kMh);
    const std::vector<k::PlaneF> metric = planes(1, 50 + side, -4.0F, 4.0F, kMw, kMh);
    const k::ref::SplatMotion motion{{flow[0], flow[1], metric[0]}};
    keep.push_back(alloc(3, kMw, kMh));
    upload(keep.back()->tensor(), flat({flow[0], flow[1], metric[0]}));
    const gpu::Tensor motion_tensor = keep.back()->tensor();
    const float s = k::gmfss_side_time(side, 0.5F);
    for (const bool high : {true, false}) {
      const int w = high ? kOw : kRw;
      const int h = high ? kOh : kRh;
      const std::vector<k::PlaneF> source = planes(3, 60 + side + (high ? 2 : 0), 0.0F, 1.0F, w, h);
      const k::MotionGrid grid = high ? high_grid : low_grid;
      auto expected = k::ref::soft_splat_at(source, motion, s, grid);
      REQUIRE(expected);
      want[static_cast<std::size_t>((high ? 0 : 2) + side)] = std::move(*expected);
      keep.push_back(alloc(3, w, h));
      upload(keep.back()->tensor(), flat(source));
      const gpu::Tensor source_tensor = keep.back()->tensor();
      keep.push_back(alloc(4, w, h));
      jobs.push_back({.source = source_tensor,
                      .motion = motion_tensor,
                      .s = s,
                      .out = keep.back()->tensor(),
                      .grid = grid,
                      .coverage = true});
    }
  }
  REQUIRE(ops.soft_splat_fill(jobs, gpu::Tensor{}, 0.5F));
  // Ordre des travaux : côté a haut, a bas, b haut, b bas.
  const std::array<std::size_t, 4> order{0, 2, 1, 3};
  for (std::size_t j = 0; j < jobs.size(); ++j) {
    CAPTURE(j);
    const k::ref::Splatted& expected = want[order[j]];
    std::vector<k::PlaneF> all = expected.planes;
    all.push_back(expected.coverage);
    // Le mouvement lu en FP32 sur la carte comme dans la référence ; une contribution qui tombe au bord d'un
    // pixel peut changer de voisin à l'arrondi près : la tolérance des splats, sur la moyenne.
    const std::vector<float> got = download(jobs[j].out);
    CHECK(worst_of(got, flat(all)) <= 0.05F);
  }
  const std::vector<k::PlaneF> fusion = planes(3, 70, 0.0F, 1.0F, 48, 32);
  auto fusion_tensor = alloc(3, 48, 32);
  upload(fusion_tensor->tensor(), flat(fusion));
  auto out = alloc(3, kOw, kOh);
  // Les sorties de la carte, relues, composées par la référence : la composition seule est jugée.
  const auto read_back = [&](std::size_t j, int w, int h) {
    const std::vector<float> data = download(jobs[j].out);
    k::ref::Splatted s{.planes = std::vector<k::PlaneF>(3, k::PlaneF(w, h)), .coverage = k::PlaneF(w, h)};
    const std::size_t plane = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
    for (std::size_t c = 0; c < 4; ++c) {
      k::PlaneF& into = c < 3 ? s.planes[c] : s.coverage;
      std::copy_n(data.begin() + static_cast<std::ptrdiff_t>(c * plane), plane, into.data.begin());
    }
    return s;
  };
  const auto composed =
      k::ref::gmfss_compose(read_back(0, kOw, kOh), read_back(2, kOw, kOh), read_back(1, kRw, kRh),
                            read_back(3, kRw, kRh), fusion, window, 0.5F);
  REQUIRE(composed);
  REQUIRE(ops.gmfss_compose({.high_a = jobs[0].out,
                             .high_b = jobs[2].out,
                             .low_a = jobs[1].out,
                             .low_b = jobs[3].out,
                             .fusion = fusion_tensor->tensor(),
                             .window = window,
                             .t = 0.5F,
                             .out = out->tensor()}));
  CHECK(worst_of(download(out->tensor()), flat(*composed)) <= k::tolerance::kRgbF16);
  // Une fenêtre qui sort de la fusion : refus.
  const auto outside = ops.gmfss_compose({.high_a = jobs[0].out,
                                          .high_b = jobs[2].out,
                                          .low_a = jobs[1].out,
                                          .low_b = jobs[3].out,
                                          .fusion = fusion_tensor->tensor(),
                                          .window = {.x = 10, .y = 10, .width = kRw, .height = kRh},
                                          .t = 0.5F,
                                          .out = out->tensor()});
  REQUIRE_FALSE(outside);
  CHECK(outside.error().code == RefusalCode::ShapeInvalid);
}
