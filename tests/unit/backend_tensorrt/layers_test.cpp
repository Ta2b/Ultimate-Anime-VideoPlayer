// La composition par calques par le backend TensorRT (KernelOps::compose_layers) contre kernels/ref : paire
// à l'instant t (bilinéaire et Catmull-Rom), grille partielle, plaque à une source avec sa propre grille, et
// les refus. Sans GPU NVIDIA, le test se SKIP.
#include "kernels/ref/layers.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "backend_fixture.h"
#include "kernels/ref/half.h"
#include "kernels/spec/layers.h"
#include "kernels/spec/tolerance.h"

namespace {

using namespace kirari;
using namespace kirari::test;
namespace k = kirari::kernels;

constexpr int kW = 64;
constexpr int kH = 48;

// Un décor texturé décalé de (ox, oy), déjà arrondi en FP16.
k::RgbImage scene(float ox, float oy) {
  k::RgbImage img(kW, kH);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < kH; ++y) {
      for (int x = 0; x < kW; ++x) {
        const float u =
            0.5F + 0.4F * std::sin(0.31F * (static_cast<float>(x) - ox) + 0.7F * static_cast<float>(c)) *
                       std::cos(0.19F * (static_cast<float>(y) - oy));
        img.planes[static_cast<std::size_t>(c)].at(x, y) = k::ref::half_to_float(k::ref::float_to_half(u));
      }
    }
  }
  return img;
}

std::vector<float> flat(const k::RgbImage& img) {
  std::vector<float> out;
  for (const k::PlaneF& p : img.planes) out.insert(out.end(), p.data.begin(), p.data.end());
  return out;
}

k::RgbImage image_of(const std::vector<float>& values) {
  k::RgbImage img(kW, kH);
  for (std::size_t c = 0; c < 3; ++c)
    std::copy_n(values.begin() + static_cast<std::ptrdiff_t>(c * kW * kH), kW * kH,
                img.planes[c].data.begin());
  return img;
}

float worst_of(const std::vector<float>& got, const std::vector<float>& expected) {
  REQUIRE(got.size() == expected.size());
  float worst = 0.0F;
  for (std::size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::fabs(got[i] - expected[i]));
  return worst;
}

}  // namespace

TEST_CASE("Backend TensorRT : calques et plaque de décor, comme la référence",
          "[backend_tensorrt][cuda][layers]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  const bool cubic = GENERATE(false, true);
  INFO("Catmull-Rom " << cubic);
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  gpu::KernelOps& ops = *loaded.backend->kernels();
  auto a = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, kH, kW}});
  auto b = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, kH, kW}});
  auto out = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, kH, kW}});
  REQUIRE((a && b && out));
  // Caméra : 6,4 pixels à droite, 2,2 vers le bas, un léger zoom ; b reçoit en plus une tache claire que la
  // garde doit écarter.
  const k::Affine camera{6.4, 1.01, 0.0, 2.2, 0.0, 1.01};
  k::RgbImage ia = scene(0.0F, 0.0F);
  k::RgbImage ib = scene(6.4F, 2.2F);
  for (int y = 30; y < 40; ++y) {
    for (int x = 40; x < 56; ++x) {
      for (auto& p : ib.planes) p.at(x, y) = 1.0F;
    }
  }
  const k::RgbImage rife = scene(3.0F, 1.0F);  // l'image de RIFE, à peu près juste
  upload((*a)->tensor(), flat(ia));
  upload((*b)->tensor(), flat(ib));

  // Paire, t = 0,3, une grille de 8 × 6 tuiles de 8 pixels dont un bloc dessin.
  const auto shifts = k::layer_shifts(camera, 0.3);
  REQUIRE(shifts);
  k::LayerParams pair{
      .grid = {.cols = 8, .rows = 6, .tile_w = 8.0F, .tile_h = 8.0F}, .count = 2, .cubic = cubic};
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 8; ++c) {
      if (!(c >= 2 && c <= 3 && r >= 1 && r <= 2)) pair.grid.set(c, r);
    }
  }
  pair.sources[0] = {.shift = k::to_float(shifts->a), .weight = 0.7F};
  pair.sources[1] = {.shift = k::to_float(shifts->b), .weight = 0.3F};
  upload((*out)->tensor(), flat(rife));
  k::RgbImage want = image_of(download((*out)->tensor()));
  REQUIRE(k::ref::compose_layers(want, image_of(download((*a)->tensor())), image_of(download((*b)->tensor())),
                                 pair));
  REQUIRE(ops.compose_layers(
      {.out = (*out)->tensor(), .a = (*a)->tensor(), .b = (*b)->tensor(), .params = pair}));
  CHECK(worst_of(download((*out)->tensor()), flat(want)) <= 2.0F * k::tolerance::kRgbF16);

  // Plaque : une source, la plaque a sa propre grille, la garde compare à l'image en place.
  const auto plate_shift = k::plate_shift(camera);
  REQUIRE(plate_shift);
  k::LayerParams plate{.grid = pair.grid,
                       .count = 1,
                       .source_grid = {.cols = 8, .rows = 6, .tile_w = 8.0F, .tile_h = 8.0F},
                       .guard_low = k::kPlateGuardLow,
                       .guard_high = k::kPlateGuardHigh,
                       .cubic = cubic};
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 7; ++c) plate.source_grid.set(c, r);
  }
  plate.sources[0] = {.shift = k::to_float(*plate_shift), .weight = 1.0F};
  upload((*out)->tensor(), flat(ib));
  k::RgbImage want_plate = image_of(download((*out)->tensor()));
  REQUIRE(k::ref::compose_layers(want_plate, image_of(download((*a)->tensor())), k::RgbImage{}, plate));
  REQUIRE(ops.compose_layers({.out = (*out)->tensor(), .a = (*a)->tensor(), .params = plate}));
  CHECK(worst_of(download((*out)->tensor()), flat(want_plate)) <= 2.0F * k::tolerance::kRgbF16);

  // Refus : la sortie est une source ; grille vide.
  const auto same = ops.compose_layers({.out = (*a)->tensor(), .a = (*a)->tensor(), .params = plate});
  REQUIRE_FALSE(same);
  CHECK(same.error().code == RefusalCode::ShapeInvalid);
  k::LayerParams empty = pair;
  empty.grid.cols = 0;
  const auto none = ops.compose_layers(
      {.out = (*out)->tensor(), .a = (*a)->tensor(), .b = (*b)->tensor(), .params = empty});
  REQUIRE_FALSE(none);
  CHECK(none.error().code == RefusalCode::FieldOutOfRange);
}
