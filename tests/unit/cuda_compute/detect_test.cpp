// Mesures d'une paire (coupe de scène, pire bloc, égalité) relues sans attente, contre
// kernels/ref/detect.h.
#include "gpu/cuda/compute/detect.h"

#include <cmath>
#include <random>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "kernels/ref/detect.h"
#include "kernels/spec/tolerance.h"
#include "rig.h"

using namespace ajp;
using namespace ajp::gpu::cuda;
using namespace ajp::gpu::cuda::test;
using kernels::Format;

namespace {

// b : a, avec un bruit d'amplitude donnée sur une zone qui coupe des blocs de 32 × 32.
kernels::YuvImage perturbed(const kernels::YuvImage& a, int amplitude, std::uint32_t seed) {
  kernels::YuvImage b = a;
  const kernels::FormatInfo info = kernels::info(a.format);
  const int peak = (1 << info.bits) - 1;
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> noise(-amplitude, amplitude);
  for (int y = a.height() / 3; y < a.height() / 2; ++y) {
    for (int x = a.width() / 5; x < a.width() * 3 / 4; ++x) {
      const int code = (b.y.at(x, y) >> info.shift) + noise(rng);
      b.y.at(x, y) = static_cast<std::uint16_t>(std::clamp(code, 0, peak) << info.shift);
    }
  }
  return b;
}

}  // namespace

TEST_CASE("les mesures de paire égalent la référence", "[cuda][detect]") {
  AJP_CUDA_RIG(rig);
  const Format format =
      GENERATE(Format::Nv12, Format::P010, Format::P016, Format::Yuv420p10, Format::Yuv444p16);
  const int amplitude = GENERATE(1, 40);
  CAPTURE(format, amplitude);
  constexpr int kW = 1000;  // ni 1000 ni 540 ne sont multiples de 32 : blocs tronqués aux bords
  constexpr int kH = 540;
  const kernels::YuvImage a = random_yuv(format, kW, kH, 89U);
  const kernels::YuvImage b = perturbed(a, amplitude, 97U);
  const DeviceYuv da = upload_yuv(rig, a);
  const DeviceYuv db = upload_yuv(rig, b);
  auto meter = PairMeter::create(*rig.pool, format, kW, kH);
  REQUIRE(meter);

  auto none = meter->previous();
  REQUIRE(none);
  CHECK_FALSE(none->has_value());  // rien de lancé

  REQUIRE(meter->measure(da.planes, db.planes, rig.s()));
  REQUIRE(rig.stream.synchronize());
  auto scores = meter->previous();
  REQUIRE(scores);
  REQUIRE(scores->has_value());
  const auto score = kernels::ref::scene_score(a.y, b.y, format, kernels::rife_geometry(kW, kH));
  const auto worst = kernels::ref::block_diff_max(a.y, b.y, format);
  REQUIRE(score);
  REQUIRE(worst);
  REQUIRE(*score > 0.0);
  CHECK(std::fabs((*scores)->scene_score - *score) <= kernels::tolerance::kSceneScoreRelative * *score);
  CHECK((*scores)->worst_block == *worst);
  CHECK_FALSE((*scores)->identical);
  auto again = meter->previous();
  REQUIRE(again);
  CHECK_FALSE(again->has_value());  // rendue une seule fois

  // La même image : score et pire bloc nuls, égalité au bit.
  REQUIRE(meter->measure(da.planes, da.planes, rig.s()));
  REQUIRE(rig.stream.synchronize());
  auto same = meter->previous();
  REQUIRE(same);
  REQUIRE(same->has_value());
  CHECK((*same)->scene_score == 0.0);
  CHECK((*same)->worst_block == 0);
  CHECK((*same)->identical);
}

TEST_CASE("les mesures de paire refusent une luma hors contrat", "[cuda][detect]") {
  AJP_CUDA_RIG(rig);
  CHECK(PairMeter::create(*rig.pool, Format::Nv12, 0, 64).error().code == RefusalCode::ShapeInvalid);
  auto meter = PairMeter::create(*rig.pool, Format::P010, 64, 32);
  REQUIRE(meter);
  const DeviceYuv good = alloc_yuv(rig, Format::P010, 64, 32);
  const DeviceYuv other_size = alloc_yuv(rig, Format::P010, 64, 34);
  const DeviceYuv other_format = alloc_yuv(rig, Format::Nv12, 64, 32);
  CHECK(meter->measure(good.planes, other_size.planes, rig.s()).error().code == RefusalCode::ShapeInvalid);
  CHECK(meter->measure(other_format.planes, good.planes, rig.s()).error().code == RefusalCode::ShapeInvalid);
  YuvPlanes null = good.planes;
  null.planes[0].ptr = 0;
  CHECK(meter->measure(good.planes, null, rig.s()).error().code == RefusalCode::InputMismatch);
}
