// GMFSS Fortuna par le backend TensorRT (gpu/kernel_ops.h) contre kernels/ref : le softmax
// splatting par additions atomiques CUDA. Sans GPU NVIDIA, le test se SKIP.
#include "kernels/ref/gmfss.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "backend_fixture.h"
#include "kernels/ref/half.h"
#include "kernels/spec/gmfss.h"
#include "kernels/spec/tolerance.h"
#include "kernels/spec/warp.h"
#include "prepare/sha256.h"

namespace {

using namespace ajp;
using namespace ajp::test;
namespace k = ajp::kernels;

constexpr int kW = 40;
constexpr int kH = 24;
constexpr int kChannels = 19;  // deux groupes de plans, le second incomplet

// Des plans dans [lo, hi], déjà arrondis en FP16 comme le tenseur les garde.
std::vector<k::PlaneF> planes_of(int w, int h, int count, int seed, float lo, float hi) {
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

std::vector<k::PlaneF> planes(int count, int seed, float lo, float hi) {
  return planes_of(kW, kH, count, seed, lo, hi);
}

std::vector<float> flat(const std::vector<k::PlaneF>& planes) {
  std::vector<float> out;
  for (const k::PlaneF& plane : planes) out.insert(out.end(), plane.data.begin(), plane.data.end());
  return out;
}

}  // namespace

TEST_CASE("Backend TensorRT : softmax splatting contre la référence", "[backend_tensorrt][cuda][gmfss]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  gpu::KernelOps& ops = *loaded.backend->kernels();
  auto alloc = [&](int channels) {
    auto t = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, channels, kH, kW}});
    REQUIRE(t);
    return std::move(*t);
  };
  const std::vector<k::PlaneF> source = planes(kChannels, 1, 0.0F, 1.0F);
  const std::vector<k::PlaneF> metric = planes(1, 3, -k::kGmfssMetricLimit, k::kGmfssMetricLimit);
  auto ts = alloc(kChannels);
  auto tm = alloc(k::kGmfssMotionChannels);
  auto out = alloc(kChannels);
  upload(ts->tensor(), flat(source));
  // ±6 px : les contributions restent dans la fenêtre partagée de leur tuile. De −14 à 22 px : la
  // fenêtre suit le déplacement moyen de la tuile, les écarts au-delà de sa marge vont directement
  // aux sommes globales, et une partie sort de l'image.
  for (const auto& [lo, hi] : {std::pair{-6.0F, 6.0F}, std::pair{-14.0F, 22.0F}}) {
    CAPTURE(lo, hi);
    const std::vector<k::PlaneF> flow = planes(2, 2, lo, hi);
    const k::ref::SplatMotion motion{{flow[0], flow[1], metric[0]}};
    upload(tm->tensor(), flat(std::vector<k::PlaneF>(motion.planes.begin(), motion.planes.end())));
    for (const float s : {0.5F, 0.25F, 1.0F}) {
      CAPTURE(s);
      REQUIRE(ops.soft_splat(ts->tensor(), tm->tensor(), s, out->tensor()));
      const auto want = k::ref::soft_splat(source, motion, s);
      REQUIRE(want);
      const std::vector<float> got = download(out->tensor());
      const std::vector<float> expected = flat(*want);
      float worst = 0.0F;
      for (std::size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::fabs(got[i] - expected[i]));
      // Seul l'ordre des additions change les arrondis (atomiques, fenêtre partagée puis globale),
      // puis le FP16 de la sortie.
      CHECK(worst <= k::tolerance::kRgbF16);
    }
  }
  REQUIRE(ops.fill(out->tensor(), 0.375F));
  for (const float v : download(out->tensor())) CHECK(v == 0.375F);
  const auto late = ops.soft_splat(ts->tensor(), tm->tensor(), 1.5F, out->tensor());
  REQUIRE_FALSE(late);
  CHECK(late.error().code == RefusalCode::FieldOutOfRange);
}

TEST_CASE("Backend TensorRT : les six splats d'un instant en un passage", "[backend_tensorrt][cuda][gmfss]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  gpu::KernelOps& ops = *loaded.backend->kernels();
  auto alloc = [&](int channels) {
    auto t = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, channels, kH, kW}});
    REQUIRE(t);
    return std::move(*t);
  };
  // Deux côtés, trois niveaux de plans différents (19 : deux groupes, le second incomplet).
  constexpr float t = 0.3F;
  constexpr std::array channels{19, 7, 3};
  std::vector<std::vector<k::PlaneF>> sources;
  std::vector<k::ref::SplatMotion> motions;
  std::vector<std::unique_ptr<gpu::TensorBuffer>> tensors;
  std::vector<gpu::KernelOps::SplatJob> jobs;
  int seed = 10;
  for (int side = 0; side < 2; ++side) {
    for (const int c : channels) {
      sources.push_back(planes(c, seed++, 0.0F, 1.0F));
      const std::vector<k::PlaneF> flow = planes(2, seed++, -4.0F, 4.0F);
      const std::vector<k::PlaneF> metric = planes(1, seed++, -k::kGmfssMetricLimit, k::kGmfssMetricLimit);
      motions.push_back({{flow[0], flow[1], metric[0]}});
      tensors.push_back(alloc(c));
      tensors.push_back(alloc(k::kGmfssMotionChannels));
      tensors.push_back(alloc(c));
      const std::size_t n = tensors.size();
      upload(tensors[n - 3]->tensor(), flat(sources.back()));
      upload(tensors[n - 2]->tensor(),
             flat(std::vector<k::PlaneF>(motions.back().planes.begin(), motions.back().planes.end())));
      jobs.push_back({.source = tensors[n - 3]->tensor(),
                      .motion = tensors[n - 2]->tensor(),
                      .s = k::gmfss_side_time(side, t),
                      .out = tensors[n - 1]->tensor()});
    }
  }
  // Les splats et l'instant de la fusion en un passage : l'instant part avec la normalisation.
  auto time = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 1, 3, 5}});
  REQUIRE(time);
  REQUIRE(ops.soft_splat_fill(jobs, (*time)->tensor(), t));
  for (const float v : download((*time)->tensor()))
    CHECK(v == k::ref::half_to_float(k::ref::float_to_half(t)));
  const auto shared = ops.soft_splat_fill(jobs, jobs[1].source, t);
  REQUIRE_FALSE(shared);
  CHECK(shared.error().code == RefusalCode::ShapeInvalid);
  for (std::size_t j = 0; j < jobs.size(); ++j) {
    CAPTURE(j);
    const auto want = k::ref::soft_splat(sources[j], motions[j], jobs[j].s);
    REQUIRE(want);
    const std::vector<float> got = download(jobs[j].out);
    const std::vector<float> expected = flat(*want);
    REQUIRE(got.size() == expected.size());
    float worst = 0.0F;
    for (std::size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::fabs(got[i] - expected[i]));
    CHECK(worst <= k::tolerance::kRgbF16);
  }
  // Au plus douze splats par passage ; un travail faux refuse le lot entier.
  std::vector<gpu::KernelOps::SplatJob> many(13, jobs[0]);
  const auto too_many = ops.soft_splat_all(many);
  REQUIRE_FALSE(too_many);
  CHECK(too_many.error().code == RefusalCode::FieldOutOfRange);
  jobs[2].out = jobs[2].motion;
  const auto shape = ops.soft_splat_all(jobs);
  REQUIRE_FALSE(shape);
  CHECK(shape.error().code == RefusalCode::ShapeInvalid);
}

TEST_CASE(
    "Backend TensorRT : le flot du niveau 1 pousse l'image de travail et l'agrandie, puis la composition",
    "[backend_tensorrt][cuda][gmfss]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  gpu::KernelOps& ops = *loaded.backend->kernels();
  // Travail 20 × 12 dans des plans remplis 32 × 16, fenêtre en (6, 2) ; flot du niveau 1 en 16 × 8 ;
  // sortie 40 × 24.
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
  int seed = 20;
  const auto alloc = [&](int channels, int width, int height) {
    auto tensor = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, channels, height, width}});
    REQUIRE(tensor);
    return std::move(*tensor);
  };
  const auto make = [&](int width, int height, int count, float lo, float hi) {
    auto tensor = alloc(count, width, height);
    std::vector<k::PlaneF> values = planes_of(width, height, count, seed++, lo, hi);
    upload(tensor->tensor(), flat(values));
    return std::pair{std::move(tensor), std::move(values)};
  };
  auto [ta, work_a] = make(w, h, 3, 0.0F, 1.0F);
  auto [tb, work_b] = make(w, h, 3, 0.0F, 1.0F);
  auto [sa, span_a] = make(ow, oh, 3, 0.0F, 1.0F);
  auto [sb, span_b] = make(ow, oh, 3, 0.0F, 1.0F);
  auto [tf, fusion] = make(fw, fh, 3, 0.0F, 1.0F);
  std::array<k::ref::SplatMotion, 2> motions;
  std::array<std::unique_ptr<gpu::TensorBuffer>, 2> tm;
  for (std::size_t side = 0; side < 2; ++side) {
    const std::vector<k::PlaneF> flow = planes_of(mw, mh, 2, seed++, -1.5F, 1.5F);
    const std::vector<k::PlaneF> metric =
        planes_of(mw, mh, 1, seed++, -k::kGmfssMetricLimit, k::kGmfssMetricLimit);
    motions[side] = {{flow[0], flow[1], metric[0]}};
    tm[side] = alloc(k::kGmfssMotionChannels, mw, mh);
    upload(tm[side]->tensor(), flat({flow[0], flow[1], metric[0]}));
  }
  std::array<std::unique_ptr<gpu::TensorBuffer>, 2> low{alloc(4, w, h), alloc(4, w, h)};
  std::array<std::unique_ptr<gpu::TensorBuffer>, 2> high{alloc(4, ow, oh), alloc(4, ow, oh)};
  const std::array<const std::vector<k::PlaneF>*, 2> works{&work_a, &work_b};
  const std::array<const std::vector<k::PlaneF>*, 2> spans{&span_a, &span_b};
  const std::array<const gpu::Tensor*, 2> work_tensors{&ta->tensor(), &tb->tensor()};
  const std::array<const gpu::Tensor*, 2> span_tensors{&sa->tensor(), &sb->tensor()};
  std::vector<gpu::KernelOps::SplatJob> jobs;
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
  REQUIRE(ops.soft_splat_all(jobs));
  const auto worst_of = [](const std::vector<float>& got, const std::vector<float>& expected) {
    REQUIRE(got.size() == expected.size());
    float worst = 0.0F;
    for (std::size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::fabs(got[i] - expected[i]));
    return worst;
  };
  const auto splatted = [&](const gpu::Tensor& tensor, int width, int height) {
    const std::vector<float> values = download(tensor);
    k::ref::Splatted out{.planes = std::vector<k::PlaneF>(3, k::PlaneF(width, height)),
                         .coverage = k::PlaneF(width, height)};
    const std::size_t area = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    for (std::size_t c = 0; c < 3; ++c)
      std::copy_n(values.begin() + static_cast<std::ptrdiff_t>(c * area), area, out.planes[c].data.begin());
    std::copy_n(values.begin() + static_cast<std::ptrdiff_t>(3 * area), area, out.coverage.data.begin());
    return out;
  };
  std::array<k::ref::Splatted, 2> got_low;
  std::array<k::ref::Splatted, 2> got_high;
  for (std::size_t side = 0; side < 2; ++side) {
    CAPTURE(side);
    const float s = k::gmfss_side_time(static_cast<int>(side), t);
    got_low[side] = splatted(low[side]->tensor(), w, h);
    got_high[side] = splatted(high[side]->tensor(), ow, oh);
    const auto want_low = k::ref::soft_splat_at(*works[side], motions[side], s, work_grid);
    const auto want_high = k::ref::soft_splat_at(*spans[side], motions[side], s, spanned_grid);
    REQUIRE((want_low && want_high));
    CHECK(worst_of(flat(got_low[side].planes), flat(want_low->planes)) <= k::tolerance::kRgbF16);
    CHECK(worst_of(got_low[side].coverage.data, want_low->coverage.data) <= k::tolerance::kRgbF16);
    CHECK(worst_of(flat(got_high[side].planes), flat(want_high->planes)) <= k::tolerance::kRgbF16);
    CHECK(worst_of(got_high[side].coverage.data, want_high->coverage.data) <= k::tolerance::kRgbF16);
  }
  auto out = alloc(3, ow, oh);
  REQUIRE(ops.gmfss_compose(high[0]->tensor(), high[1]->tensor(), low[0]->tensor(), low[1]->tensor(),
                            tf->tensor(), window, t, out->tensor()));
  const auto want =
      k::ref::gmfss_compose(got_high[0], got_high[1], got_low[0], got_low[1], fusion, window, t);
  REQUIRE(want);
  CHECK(worst_of(download(out->tensor()), flat(*want)) <= k::tolerance::kRgbF16);
  const auto outside =
      ops.gmfss_compose(high[0]->tensor(), high[1]->tensor(), low[0]->tensor(), low[1]->tensor(),
                        tf->tensor(), k::Window{.x = 14, .y = 2, .width = w, .height = h}, t, out->tensor());
  REQUIRE_FALSE(outside);
  CHECK(outside.error().code == RefusalCode::ShapeInvalid);
}

TEST_CASE("Backend TensorRT : composition par objets texture, deux pixels par fil, contre la référence",
          "[backend_tensorrt][cuda][gmfss]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  // Travail 32 × 16, fusion 48 × 24 (fenêtre en (8, 4)), sortie 64 × 32 : largeurs de 64, 96 et
  // 128 octets, au pas des objets texture. À l'échelle 2 la fraction vaut 1/4 ou 3/4, que le filtre
  // matériel rend exactement ; la marge de kWarpFiltered couvre sa précision interne.
  constexpr int w = 32;
  constexpr int h = 16;
  constexpr k::Window window{.x = 8, .y = 4, .width = w, .height = h};
  constexpr float t = 0.3F;
  const auto splatted = [](int width, int height, int seed) {
    return k::ref::Splatted{.planes = planes_of(width, height, 3, seed, 0.0F, 1.0F),
                            .coverage = planes_of(width, height, 1, seed + 1, 0.0F, 1.0F)[0]};
  };
  const auto tensor_of = [&](std::vector<k::PlaneF> values, const k::PlaneF* coverage) {
    if (coverage != nullptr) values.push_back(*coverage);
    auto tensor = loaded.backend->alloc_tensor(
        gpu::TensorType::F16,
        {.dims = {1, static_cast<std::int64_t>(values.size()), values[0].height, values[0].width}});
    REQUIRE(tensor);
    upload((*tensor)->tensor(), flat(values));
    return std::move(*tensor);
  };
  const std::array<k::ref::Splatted, 2> high{splatted(2 * w, 2 * h, 3), splatted(2 * w, 2 * h, 5)};
  const std::array<k::ref::Splatted, 2> low{splatted(w, h, 7), splatted(w, h, 9)};
  const std::vector<k::PlaneF> fusion = planes_of(48, 24, 3, 11, 0.0F, 1.0F);
  const auto ha = tensor_of(high[0].planes, &high[0].coverage);
  const auto hb = tensor_of(high[1].planes, &high[1].coverage);
  const auto la = tensor_of(low[0].planes, &low[0].coverage);
  const auto lb = tensor_of(low[1].planes, &low[1].coverage);
  const auto tf = tensor_of(fusion, nullptr);
  auto out = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, 2 * h, 2 * w}});
  REQUIRE(out);
  REQUIRE(loaded.backend->kernels()->gmfss_compose(ha->tensor(), hb->tensor(), la->tensor(), lb->tensor(),
                                                   tf->tensor(), window, t, (*out)->tensor()));
  const auto want = k::ref::gmfss_compose(high[0], high[1], low[0], low[1], fusion, window, t);
  REQUIRE(want);
  const std::vector<float> got = download((*out)->tensor());
  const std::vector<float> expected = flat(*want);
  REQUIRE(got.size() == expected.size());
  float worst = 0.0F;
  for (std::size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::fabs(got[i] - expected[i]));
  CHECK(worst <= k::tolerance::kWarpFiltered);
}

TEST_CASE("Backend TensorRT : le niveau 1 en deux splats, chacun dans sa tranche de warped",
          "[backend_tensorrt][cuda][gmfss]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  gpu::KernelOps& ops = *loaded.backend->kernels();
  // Export de version 2 : l'image (3 plans) et les caractéristiques (ici 5, 64 dans GMFSS) arrivent en deux
  // tenseurs ; chaque splat écrit sa tranche de la sortie à 3 + 5 plans, comme le splat de leur
  // concaténation. 32 × 16 : la tranche des caractéristiques commence à 3 072 octets, multiple de 256.
  constexpr int w = 32;
  constexpr int h = 16;
  constexpr int image_planes = 3;
  constexpr int feature_planes = 5;
  const auto alloc = [&](int channels) {
    auto tensor = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, channels, h, w}});
    REQUIRE(tensor);
    return std::move(*tensor);
  };
  const std::vector<k::PlaneF> image = planes_of(w, h, image_planes, 31, 0.0F, 1.0F);
  const std::vector<k::PlaneF> features = planes_of(w, h, feature_planes, 32, -2.0F, 2.0F);
  const std::vector<k::PlaneF> flow = planes_of(w, h, 2, 33, -5.0F, 5.0F);
  const std::vector<k::PlaneF> metric = planes_of(w, h, 1, 34, -k::kGmfssMetricLimit, k::kGmfssMetricLimit);
  const k::ref::SplatMotion motion{{flow[0], flow[1], metric[0]}};
  auto t_image = alloc(image_planes);
  auto t_features = alloc(feature_planes);
  auto t_motion = alloc(k::kGmfssMotionChannels);
  auto whole = alloc(image_planes + feature_planes);
  upload(t_image->tensor(), flat(image));
  upload(t_features->tensor(), flat(features));
  upload(t_motion->tensor(), flat(std::vector<k::PlaneF>(motion.planes.begin(), motion.planes.end())));
  const auto slice = [&](int first, int count) {
    gpu::Tensor part = whole->tensor();
    part.shape.dims[1] = count;
    part.offset += static_cast<std::uint64_t>(first) * w * h * sizeof(std::uint16_t);
    return part;
  };
  std::vector<k::PlaneF> joined = image;
  joined.insert(joined.end(), features.begin(), features.end());
  for (const float s : {0.3F, 1.0F}) {
    CAPTURE(s);
    const std::array jobs{
        gpu::KernelOps::SplatJob{
            .source = t_image->tensor(), .motion = t_motion->tensor(), .s = s, .out = slice(0, image_planes)},
        gpu::KernelOps::SplatJob{.source = t_features->tensor(),
                                 .motion = t_motion->tensor(),
                                 .s = s,
                                 .out = slice(image_planes, feature_planes)}};
    REQUIRE(ops.fill(whole->tensor(), 7.0F));  // une tranche non écrite resterait à 7
    REQUIRE(ops.soft_splat_all(jobs));
    const auto want = k::ref::soft_splat(joined, motion, s);
    REQUIRE(want);
    const std::vector<float> got = download(whole->tensor());
    const std::vector<float> expected = flat(*want);
    REQUIRE(got.size() == expected.size());
    float worst = 0.0F;
    for (std::size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::fabs(got[i] - expected[i]));
    CHECK(worst <= k::tolerance::kRgbF16);
  }
}

// Caché : `ajp_backend_tensorrt_tests "[.gmfss_split]"`, sous `ajp-gpu`. Les trois graphes (pair d'avant en
// 67 canaux, pair d'après sans Concat, fusion), aux tailles fixées, sont ceux des variables CMake
// AJP_GMFSS_SPLIT_{OLD_PAIR,NEW_PAIR,FUSION} (tests/unit/backend_tensorrt/CMakeLists.txt) ; l'un absent,
// SKIP.
TEST_CASE("Backend TensorRT : la fusion rend la même image avec la paire sans Concat",
          "[.gmfss_split][backend_tensorrt][cuda][gmfss]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  const std::array<std::filesystem::path, 3> files{AJP_GMFSS_SPLIT_OLD_PAIR, AJP_GMFSS_SPLIT_NEW_PAIR,
                                                   AJP_GMFSS_SPLIT_FUSION};
  for (const std::filesystem::path& file : files) {
    if (!std::filesystem::is_regular_file(file)) SKIP("graphe GMFSS absent : '" + file.string() + "'");
  }
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  gpu::KernelOps& ops = *loaded.backend->kernels();
  constexpr int width = 1280;
  constexpr int height = 768;
  const auto engine = [&](const std::filesystem::path& file, const gpu::Shape& shape) {
    gpu::Progress progress;
    const auto sha256 = prepare::sha256_file(file);
    REQUIRE(sha256);
    const gpu::ModelSpec spec{.id = file.stem().string(), .onnx_path = file, .sha256 = *sha256};
    auto artifact = loaded.backend->build(spec, {.min = shape, .opt = shape, .max = shape}, progress);
    REQUIRE(artifact);
    auto loaded_engine = loaded.backend->load(*artifact);
    REQUIRE(loaded_engine);
    return std::move(*loaded_engine);
  };
  const auto old_pair = engine(files[0], {.dims = {1, k::kGmfssFramePlanes, height, width}});
  const auto new_pair = engine(files[1], {.dims = {1, k::kGmfssFramePlanes, height, width}});
  const auto fusion = engine(files[2], {.dims = {1, 3, height / 2, width / 2}});

  // Deux images lisses décalées l'une de l'autre : un vrai mouvement.
  auto frames = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 6, height, width}});
  REQUIRE(frames);
  constexpr std::size_t kRows = height;
  constexpr std::size_t kColumns = width;
  std::vector<float> pixels(std::size_t{6} * kRows * kColumns);
  for (int c = 0; c < 6; ++c) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const float dx = c < 3 ? 0.0F : 5.0F;
        const float dy = c < 3 ? 0.0F : 3.0F;
        const float u =
            0.5F +
            0.25F * std::sin(0.021F * (static_cast<float>(x) + dx) + 0.7F * static_cast<float>(c % 3)) +
            0.25F * std::cos(0.017F * (static_cast<float>(y) + dy));
        pixels[(static_cast<std::size_t>(c) * kRows + static_cast<std::size_t>(y)) * kColumns +
               static_cast<std::size_t>(x)] = u;
      }
    }
  }
  upload((*frames)->tensor(), pixels);

  // Les tenseurs d'un graphe selon le nom de ses sorties ; les mêmes noms à l'ancienne et à la nouvelle
  // paire.
  std::vector<std::pair<std::string, std::unique_ptr<gpu::TensorBuffer>>> tensors;
  const auto shape_of = [&](const std::string& name) {
    int level = 0;
    int channels = 0;
    if (name.starts_with("half")) {
      level = 1;
      channels = 3;
    } else {
      level = name.back() - '0';
      channels = name.starts_with("motion")    ? k::kGmfssMotionChannels
                 : name.starts_with("feature") ? 64
                                               : k::gmfss_source_channels(level);
    }
    return gpu::Shape{.dims = {1, channels, height >> level, width >> level}};
  };
  const auto tensor = [&](const std::string& name, gpu::Shape shape) -> const gpu::Tensor& {
    for (const auto& [known, buffer] : tensors) {
      if (known == name) return buffer->tensor();
    }
    auto buffer = loaded.backend->alloc_tensor(gpu::TensorType::F16, shape);
    REQUIRE(buffer);
    tensors.emplace_back(name, std::move(*buffer));
    return tensors.back().second->tensor();
  };
  // Exécute une paire, ses six ou huit splats puis la fusion : l'image de sortie.
  const auto render = [&](gpu::Engine& pair, const std::string& tag) {
    std::vector<gpu::Tensor> outputs;
    for (const std::string& name : pair.output_names()) outputs.push_back(tensor(tag + name, shape_of(name)));
    const std::array inputs{(*frames)->tensor()};
    REQUIRE(loaded.backend->enqueue(pair, inputs, outputs));
    const bool split = std::ranges::contains(pair.output_names(), std::string("feature0_1"));
    std::vector<gpu::KernelOps::SplatJob> jobs;
    for (int side = 0; side < 2; ++side) {
      for (int level = 1; level <= k::kGmfssLevels; ++level) {
        const gpu::Shape warped_shape{
            .dims = {1, k::gmfss_source_channels(level), height >> level, width >> level}};
        const gpu::Tensor& warped = tensor(std::format("{}warped{}_{}", tag, side, level), warped_shape);
        const gpu::Tensor& motion =
            tensor(std::format("{}motion{}_{}", tag, side, level), shape_of("motion0_1"));
        const float s = k::gmfss_side_time(side, 0.5F);
        if (level == 1 && split) {
          const std::int64_t plane = static_cast<std::int64_t>(height / 2) * (width / 2);
          gpu::Tensor image_part = warped;
          image_part.shape.dims[1] = 3;
          gpu::Tensor feature_part = warped;
          feature_part.shape.dims[1] = 64;
          feature_part.offset += static_cast<std::uint64_t>(3 * plane) * 2;
          jobs.push_back({.source = tensor(std::format("{}half{}", tag, side), shape_of("half0")),
                          .motion = motion,
                          .s = s,
                          .out = image_part});
          jobs.push_back({.source = tensor(std::format("{}feature{}_1", tag, side), shape_of("feature0_1")),
                          .motion = motion,
                          .s = s,
                          .out = feature_part});
        } else {
          jobs.push_back(
              {.source = tensor(std::format("{}source{}_{}", tag, side, level), shape_of("source0_2")),
               .motion = motion,
               .s = s,
               .out = warped});
        }
      }
    }
    REQUIRE(ops.soft_splat_all(jobs));
    std::vector<gpu::Tensor> inputs_of_fusion;
    for (const std::string& name : fusion->input_names()) {
      inputs_of_fusion.push_back(
          name.starts_with("half")
              ? tensor(tag + name, shape_of(name))
              : tensor(tag + name,
                       gpu::Shape{.dims = {1, k::gmfss_source_channels(name.back() - '0'),
                                           height >> (name.back() - '0'), width >> (name.back() - '0')}}));
    }
    auto image = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, height, width}});
    REQUIRE(image);
    std::array results{(*image)->tensor()};
    REQUIRE(loaded.backend->enqueue(*fusion, inputs_of_fusion, results));
    return download((*image)->tensor());
  };
  // Les tenseurs des noms de sortie de l'ancienne paire sont ceux de la nouvelle à une exception près : les
  // noms portent chacun leur préfixe (« old/ », « new/ », « again/ »).
  const std::vector<float> before = render(*old_pair, "old/");
  const std::vector<float> before_again =
      render(*old_pair, "old2/");  // le bruit de l'ordre des additions atomiques
  const std::vector<float> after = render(*new_pair, "new/");
  const std::vector<float> after_again = render(*new_pair, "new2/");
  REQUIRE(before.size() == after.size());
  const auto compare = [](const std::vector<float>& a, const std::vector<float>& b) {
    REQUIRE(a.size() == b.size());
    double squares = 0.0;
    float worst = 0.0F;
    for (std::size_t i = 0; i < a.size(); ++i) {
      worst = std::max(worst, std::fabs(a[i] - b[i]));
      squares += static_cast<double>(a[i] - b[i]) * static_cast<double>(a[i] - b[i]);
    }
    const double mse = squares / static_cast<double>(a.size());
    return std::pair{worst, mse > 0.0 ? 10.0 * std::log10(1.0 / mse) : 200.0};
  };
  // La paire elle-même : les sorties communes (mouvement, images réduites, niveaux 2 et 3) et les 64
  // caractéristiques de feature{k}_1 contre les plans 3 à 66 de source{k}_1 sont celles de l'ancien graphe.
  float pair_worst = 0.0F;
  for (int side = 0; side < 2; ++side) {
    for (const std::string& name :
         {std::format("motion{}_1", side), std::format("motion{}_2", side), std::format("motion{}_3", side),
          std::format("half{}", side), std::format("source{}_2", side), std::format("source{}_3", side)}) {
      const std::vector<float> a = download(tensor("old/" + name, shape_of(name)));
      const std::vector<float> b = download(tensor("new/" + name, shape_of(name)));
      pair_worst = std::max(pair_worst, compare(a, b).first);
    }
    const std::vector<float> whole =
        download(tensor(std::format("old/source{}_1", side), shape_of("source0_1")));
    const std::vector<float> features =
        download(tensor(std::format("new/feature{}_1", side), shape_of("feature0_1")));
    const std::vector<float> tail(whole.end() - static_cast<std::ptrdiff_t>(features.size()), whole.end());
    pair_worst = std::max(pair_worst, compare(tail, features).first);
  }
  const auto [old_worst, old_psnr] = compare(before, before_again);
  const auto [new_worst, new_psnr] = compare(after, after_again);
  const auto [worst, psnr] = compare(before, after);
  INFO("sorties de la paire, avant contre après : pire écart " << pair_worst);
  INFO("avant contre avant : pire pixel " << old_worst << ", PSNR " << old_psnr << " dB");
  INFO("après contre après : pire pixel " << new_worst << ", PSNR " << new_psnr << " dB");
  INFO("avant contre après : pire pixel " << worst << ", PSNR " << psnr << " dB");
  // La paire rend les mêmes tenseurs ; ensuite seul l'ordre des additions atomiques du splat bouge l'image
  // (le bruit entre deux exécutions du même graphe, 70 dB), à 1 ulp du FP16 près (2^-10 à 1,0).
  CHECK(pair_worst <= 1.0F / 1024.0F);
  CHECK(worst <= std::max(old_worst, new_worst) + 1.0F / 1024.0F);
  CHECK(psnr >= std::min(old_psnr, new_psnr) - 3.0);
}
