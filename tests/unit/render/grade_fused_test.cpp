// Le corps fusionné du grade (trois passes, scène donnée par le lecteur) contre l'ancien corps (neuf passes,
// scène mesurée dans le shader) : sans adaptation ils rendent la même image ; l'APL reçue pilote le
// limiteur des scènes claires.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "grade_gpu_support.h"
#include "render/grade.h"

namespace ajp::render {
using namespace gpu_test;

namespace {

// Un pan sombre, un pan clair et un éclat isolé : la carte locale et l'isolement jouent tous deux.
std::vector<std::uint16_t> mixed() {
  std::vector<std::uint16_t> pixels(static_cast<std::size_t>(kSide) * kSide * 4);
  for (int y = 0; y < kSide; ++y)
    for (int x = 0; x < kSide; ++x) {
      double nits = x < kSide / 2 ? 20.0 : 250.0;
      if (x >= 60 && x < 63 && y >= 60 && y < 63) nits = 800.0;
      const auto at = (static_cast<std::size_t>(y) * kSide + static_cast<std::size_t>(x)) * 4;
      for (std::size_t c = 0; c < 3; ++c)
        pixels[at + c] = static_cast<std::uint16_t>(std::lround(nits_to_pq(nits) * 65535.0));
      pixels[at + 3] = 65535;
    }
  return pixels;
}

GradeParams fixed_look() {
  GradeParams p = strong(0.5);
  p.area_gain = 2.0;
  p.spec_gain = 3.0;
  return p;
}

}  // namespace

TEST_CASE("grade fusionné : sans adaptation, la même image que l'ancien corps", "[render][gpu]") {
  Gpu fused(GradeBody::Fused);
  require_device(fused);
  REQUIRE(fused.grade);
  Gpu legacy(GradeBody::Legacy);
  REQUIRE(legacy.grade);
  const std::vector<Rgb> a = graded_rgb(fused, mixed(), fixed_look());
  const std::vector<Rgb> b = graded_rgb(legacy, mixed(), fixed_look());
  REQUIRE(a.size() == b.size());
  for (std::size_t x = 0; x < a.size(); ++x)
    for (std::size_t c = 0; c < 3; ++c) CHECK(std::abs(a[x][c] - b[x][c]) <= 0.01 * b[x][c] + 0.05);
}

TEST_CASE("grade fusionné : l'APL donnée par le lecteur pilote le limiteur des scènes claires",
          "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  GradeParams params = fixed_look();
  params.abl = 0.5;
  const auto center = [&](double apl) {
    params.fs_apl = apl;
    return graded(gpu, uniform(400.0), params)[kSide / 2];
  };
  const double unknown = center(-1.0);
  CHECK(std::abs(center(20.0) - unknown) < 1e-3 * unknown);  // scène sombre : rien ne bouge
  CHECK(center(400.0) < 0.9 * unknown);                      // scène claire : limitée
  CHECK(center(400.0) > 390.0);                              // jamais sous la source
}

}  // namespace ajp::render
