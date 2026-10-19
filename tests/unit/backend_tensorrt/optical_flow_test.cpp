// Le flux optique matériel (NVIDIA Optical Flow) par le backend TensorRT, sur la RTX : une image
// texturée décalée de 6 pixels donne un mouvement de 6 pixels, et le warp place l'image du milieu
// à mi-chemin. Sans GPU NVIDIA, ou sans flux optique dans le pilote, chaque test se SKIP.
#include "kernels/spec/optical_flow.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "backend_fixture.h"

namespace {

using namespace ajp;
using namespace ajp::test;

constexpr int kW = 320;
constexpr int kH = 192;
constexpr int kShift = 6;  // pixels vers la droite, de a à b

// Texture lisse et sans période courte (somme de sinus), assez riche pour le flux optique ; x en
// pixels, décalée de `shift` vers la droite.
std::vector<float> texture(int shift) {
  std::vector<float> v(static_cast<std::size_t>(3 * kW * kH));
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < kH; ++y) {
      for (int x = 0; x < kW; ++x) {
        const float u = static_cast<float>(x - shift);
        const float w = static_cast<float>(y);
        v[static_cast<std::size_t>((c * kH + y) * kW + x)] = 0.5F + 0.2F * std::sin(0.31F * u + 0.17F * w) +
                                                             0.15F * std::sin(0.07F * u - 0.23F * w + 1.0F) +
                                                             0.1F * std::cos(0.27F * u + 0.41F * w);
      }
    }
  }
  return v;
}

}  // namespace

TEST_CASE("Backend TensorRT : flux optique matériel, une translation de 6 pixels",
          "[backend_tensorrt][cuda][nvidia]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  gpu::KernelOps& ops = *loaded.backend->kernels();
  auto grid = ops.open_optical_flow(kW, kH);
  if (!grid && grid.error().code == RefusalCode::LibraryUnavailable) SKIP(grid.error().detail);
  REQUIRE(grid);
  CHECK((*grid == 2 || *grid == 4));
  const int cx = kernels::flow_cells(kW, *grid);
  const int cy = kernels::flow_cells(kH, *grid);
  auto alloc = [&](std::int64_t c, std::int64_t h, std::int64_t w) {
    auto t = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, c, h, w}});
    REQUIRE(t);
    return std::move(*t);
  };
  auto a = alloc(3, kH, kW);
  auto b = alloc(3, kH, kW);
  auto flow = alloc(4, cy, cx);
  auto mask = alloc(1, cy, cx);
  auto middle = alloc(3, kH, kW);
  upload(a->tensor(), texture(0));
  upload(b->tensor(), texture(kShift));
  const gpu::Motion motion{.flow = flow->tensor(),
                           .mask = mask->tensor(),
                           .area = kernels::Window{.x = 0, .y = 0, .width = cx, .height = cy}};
  // Mouvement d'une autre taille : refusé par son nom.
  const gpu::Motion small{.flow = flow->tensor(),
                          .mask = mask->tensor(),
                          .area = kernels::Window{.x = 0, .y = 0, .width = cx - 1, .height = cy}};
  CHECK(ops.optical_flow(a->tensor(), b->tensor(), small).error().code == RefusalCode::ShapeInvalid);
  // Deux paires : la seconde profite des indices temporels de la première.
  REQUIRE(ops.optical_flow(a->tensor(), b->tensor(), motion));
  REQUIRE(ops.optical_flow(a->tensor(), b->tensor(), motion));
  const std::vector<float> f = download(flow->tensor());
  const std::vector<float> m = download(mask->tensor());
  // Au cœur de l'image (le bord manque de texture d'un côté) : f1x × grille ≈ 3 pixels, f0x ≈ −3,
  // aucun mouvement vertical, masque 1/2.
  const std::size_t plane = static_cast<std::size_t>(cx) * static_cast<std::size_t>(cy);
  std::vector<float> half_shift;
  for (int j = cy / 4; j < 3 * cy / 4; ++j) {
    for (int i = cx / 4; i < 3 * cx / 4; ++i) {
      const std::size_t o =
          static_cast<std::size_t>(j) * static_cast<std::size_t>(cx) + static_cast<std::size_t>(i);
      half_shift.push_back(f[2 * plane + o] * static_cast<float>(*grid));
      CHECK(std::abs(f[o] + f[2 * plane + o]) < 0.05F);
      CHECK(std::abs(f[3 * plane + o] * static_cast<float>(*grid)) < 0.75F);
      CHECK(std::abs(m[o] - 0.5F) < 0.2F);
    }
  }
  std::ranges::sort(half_shift);
  const float median = half_shift[half_shift.size() / 2];
  CHECK(std::abs(median - kShift / 2.0F) < 0.5F);
  const auto close =
      std::ranges::count_if(half_shift, [](float v) { return std::abs(v - kShift / 2.0F) < 1.0F; });
  CHECK(close * 10 >= static_cast<std::ptrdiff_t>(half_shift.size()) * 9);
  // Le warp au milieu : l'image décalée de 3 pixels, au cœur.
  REQUIRE(ops.warp_blend(a->tensor(), b->tensor(), motion, 0.5F, middle->tensor()));
  const std::vector<float> got = download(middle->tensor());
  const std::vector<float> want = texture(kShift / 2);
  double error = 0.0;
  int count = 0;
  for (int y = kH / 4; y < 3 * kH / 4; ++y) {
    for (int x = kW / 4; x < 3 * kW / 4; ++x) {
      const std::size_t i = static_cast<std::size_t>(y * kW + x);
      error += std::abs(got[i] - want[i]);
      ++count;
    }
  }
  CHECK(error / count < 0.02);
}

TEST_CASE("Backend TensorRT : flux optique refusé par son nom tant qu'il n'est pas ouvert",
          "[backend_tensorrt][cuda]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  gpu::KernelOps& ops = *loaded.backend->kernels();
  auto a = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, 64, 64}});
  REQUIRE(a);
  const gpu::Motion motion{.flow = (*a)->tensor(), .mask = (*a)->tensor(), .area = {}};
  CHECK(ops.optical_flow((*a)->tensor(), (*a)->tensor(), motion).error().code == RefusalCode::FieldMissing);
  CHECK(ops.open_optical_flow(16, 16).error().code == RefusalCode::ShapeInvalid);
}
