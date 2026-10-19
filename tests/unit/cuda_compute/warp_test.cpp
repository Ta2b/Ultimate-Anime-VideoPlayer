// Warp arrière et mélange, et plus grand déplacement relu sans attente, contre
// kernels/ref/warp.h.
#include "gpu/cuda/compute/warp.h"

#include <optional>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "kernels/ref/half.h"
#include "kernels/ref/warp.h"
#include "kernels/spec/tolerance.h"
#include "rig.h"

using namespace ajp;
using namespace ajp::gpu::cuda;
using namespace ajp::gpu::cuda::test;

namespace {

// Le mouvement de RIFE dans ses plans remplis 128 × 64, fenêtre 100 × 50 en (14, 6).
constexpr kernels::Window kArea{.x = 14, .y = 6, .width = 100, .height = 50};

kernels::ref::Motion random_motion(bool half) {
  kernels::ref::Motion m;
  for (std::size_t c = 0; c < 4; ++c)
    m.flow[c] = random_plane(128, 64, -6.0F, 6.0F, 61U + static_cast<std::uint32_t>(c));
  m.mask = random_plane(128, 64, 0.0F, 1.0F, 67U);
  m.area = kArea;
  if (half) {
    for (kernels::PlaneF* p : {&m.flow[0], &m.flow[1], &m.flow[2], &m.flow[3], &m.mask}) {
      for (float& v : p->data) v = kernels::ref::round_to_half(v);
    }
  }
  return m;
}

struct DeviceMotion {
  DeviceTensor flow;
  DeviceTensor mask;
  MotionView view;
};

DeviceMotion upload_motion(Rig& rig, const kernels::ref::Motion& m, Elem elem) {
  DeviceMotion out{.flow = upload_planes(rig, {m.flow.begin(), m.flow.end()}, elem),
                   .mask = upload_planes(rig, {m.mask}, elem),
                   .view = {}};
  out.view = {.flow = out.flow.tensor, .mask = out.mask.tensor, .area = m.area};
  return out;
}

}  // namespace

TEST_CASE("le warp au milieu de la paire égale la référence au bit", "[cuda][warp]") {
  AJP_CUDA_RIG(rig);
  const Elem elem = GENERATE(Elem::F32, Elem::F16);
  // Sortie à la taille de la fenêtre, puis deux fois plus grande (S3 : warp à la sortie).
  const int scale = GENERATE(1, 2);
  CAPTURE(elem, scale);
  const bool half = elem == Elem::F16;
  const int w = kArea.width * scale;
  const int h = kArea.height * scale;
  kernels::RgbImage a = random_rgb(w, h, 0.0F, 1.0F, 71U);
  kernels::RgbImage b = random_rgb(w, h, 0.0F, 1.0F, 73U);
  if (half) {
    a = kernels::ref::round_to_half(a);
    b = kernels::ref::round_to_half(b);
  }
  const kernels::ref::Motion motion = random_motion(half);
  const DeviceTensor da = upload_rgb(rig, a, elem);
  const DeviceTensor db = upload_rgb(rig, b, elem);
  const DeviceMotion dm = upload_motion(rig, motion, elem);
  DeviceTensor out = alloc_tensor(rig, elem, w, h);
  REQUIRE(warp_blend(da.tensor, db.tensor, dm.view, 0.5F, out.tensor, rig.s()));
  auto expected = kernels::ref::warp_blend(a, b, motion);
  REQUIRE(expected);
  if (half) expected = kernels::ref::round_to_half(*expected);
  CHECK(max_diff(download_rgb(rig, out), *expected) == 0.0F);
}

TEST_CASE("le warp à un autre instant reste dans la tolérance kWarp", "[cuda][warp]") {
  AJP_CUDA_RIG(rig);
  const float t = GENERATE(0.125F, 0.25F, 0.75F, 0.9F);
  const Elem elem = GENERATE(Elem::F32, Elem::F16);
  CAPTURE(t, elem);
  const bool half = elem == Elem::F16;
  kernels::RgbImage a = random_rgb(kArea.width, kArea.height, 0.0F, 1.0F, 79U);
  kernels::RgbImage b = random_rgb(kArea.width, kArea.height, 0.0F, 1.0F, 83U);
  if (half) {
    a = kernels::ref::round_to_half(a);
    b = kernels::ref::round_to_half(b);
  }
  const kernels::ref::Motion motion = random_motion(half);
  const DeviceTensor da = upload_rgb(rig, a, elem);
  const DeviceTensor db = upload_rgb(rig, b, elem);
  const DeviceMotion dm = upload_motion(rig, motion, elem);
  DeviceTensor out = alloc_tensor(rig, elem, kArea.width, kArea.height);
  REQUIRE(warp_blend(da.tensor, db.tensor, dm.view, t, out.tensor, rig.s()));
  auto expected = kernels::ref::warp_blend_at(a, b, motion, t);
  REQUIRE(expected);
  if (half) expected = kernels::ref::round_to_half(*expected);
  // logf et expf de CUDA contre ceux de la bibliothèque C : quelques ulp sur le poids du
  // mélange, qui se propagent à la valeur.
  const float diff = max_diff(download_rgb(rig, out), *expected);
  INFO("écart " << diff);
  CHECK(diff <= kernels::tolerance::kWarp);
}

TEST_CASE("le plus grand déplacement se relit sans attente, une fois", "[cuda][warp]") {
  AJP_CUDA_RIG(rig);
  const Elem elem = GENERATE(Elem::F32, Elem::F16);
  const kernels::ref::Motion motion = random_motion(elem == Elem::F16);
  const DeviceMotion dm = upload_motion(rig, motion, elem);
  auto meter = FlowMax::create(*rig.pool);
  REQUIRE(meter);
  auto none = meter->poll();
  REQUIRE(none);
  CHECK_FALSE(none->has_value());  // rien de lancé

  REQUIRE(meter->launch(dm.view, rig.s()));
  REQUIRE(rig.stream.synchronize());
  auto value = meter->poll();
  REQUIRE(value);
  REQUIRE(value->has_value());
  const auto expected = kernels::ref::flow_max(motion);
  REQUIRE(expected);
  CHECK(**value == *expected);
  auto again = meter->poll();
  REQUIRE(again);
  CHECK_FALSE(again->has_value());  // rendu une seule fois

  // Un mouvement nul : 0, pas une valeur restée de la mesure précédente.
  kernels::ref::Motion still = motion;
  for (kernels::PlaneF& p : still.flow) p = kernels::PlaneF(128, 64, 0.0F);
  const DeviceMotion ds = upload_motion(rig, still, elem);
  REQUIRE(meter->launch(ds.view, rig.s()));
  REQUIRE(rig.stream.synchronize());
  auto zero = meter->poll();
  REQUIRE(zero);
  REQUIRE(zero->has_value());
  CHECK(**zero == 0.0F);
}

TEST_CASE("le warp refuse les instants et mouvements hors contrat", "[cuda][warp]") {
  AJP_CUDA_RIG(rig);
  const kernels::ref::Motion motion = random_motion(true);
  const DeviceMotion dm = upload_motion(rig, motion, Elem::F16);
  DeviceTensor a = alloc_tensor(rig, Elem::F16, 100, 50);
  DeviceTensor out = alloc_tensor(rig, Elem::F16, 100, 50);
  for (const float t : {0.0F, 1.0F, -0.5F}) {
    CHECK(warp_blend(a.tensor, a.tensor, dm.view, t, out.tensor, rig.s()).error().code ==
          RefusalCode::FieldOutOfRange);
  }
  MotionView outside = dm.view;
  outside.area.x = 30;  // 30 + 100 > 128
  CHECK(warp_blend(a.tensor, a.tensor, outside, 0.5F, out.tensor, rig.s()).error().code ==
        RefusalCode::ShapeInvalid);
  DeviceTensor f32 = alloc_tensor(rig, Elem::F32, 100, 50);
  CHECK(warp_blend(a.tensor, f32.tensor, dm.view, 0.5F, out.tensor, rig.s()).error().code ==
        RefusalCode::FormatUnsupported);
  DeviceTensor small = alloc_tensor(rig, Elem::F16, 50, 50);
  CHECK(warp_blend(a.tensor, small.tensor, dm.view, 0.5F, out.tensor, rig.s()).error().code ==
        RefusalCode::ShapeInvalid);
}
