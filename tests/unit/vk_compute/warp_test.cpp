// Warp arrière par flot et mélange à l'instant t contre ref::warp_blend et ref::warp_blend_at,
// mouvement dans une fenêtre de plans remplis, en float et en float16 (tolérance kWarp).
#include "kernels/ref/warp.h"

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "kernels/ref/half.h"
#include "kernels/spec.h"
#include "rig.h"

using namespace ajp;
using namespace ajp::gpu::vk;
using namespace ajp::gpu::vk::test;
namespace k = ajp::kernels;

namespace {

k::PlaneF stored(const k::PlaneF& plane, Precision p) {
  if (p == Precision::F32) return plane;
  k::PlaneF out = plane;
  for (float& v : out.data) v = k::ref::round_to_half(v);
  return out;
}

// Mouvement de plans fw × fh, fenêtre area ; hors fenêtre, des valeurs qui ne doivent pas compter.
k::ref::Motion synthetic_motion(int fw, int fh, k::Window area, float reach, Precision p) {
  k::ref::Motion m{.flow = {}, .mask = stored(synthetic_plane(fw, fh, 40, 0.0F, 1.0F), p), .area = area};
  for (std::size_t c = 0; c < m.flow.size(); ++c) {
    m.flow[c] = stored(synthetic_plane(fw, fh, 41 + static_cast<std::uint32_t>(c), -reach, reach), p);
  }
  return m;
}

struct Pair {
  k::RgbImage a;
  k::RgbImage b;
  k::ref::Motion motion;
};

Pair synthetic_pair(int w, int h, int fw, int fh, k::Window area, Precision p) {
  return {.a = as_stored(synthetic_rgb(w, h, 50, 0.0F, 1.0F), p),
          .b = as_stored(synthetic_rgb(w, h, 51, 0.0F, 1.0F), p),
          .motion = synthetic_motion(fw, fh, area, 3.0F, p)};
}

void check_warp(Rig& rig, const Pair& pair, float t, Precision p) {
  INFO("t = " << t << ", précision " << static_cast<int>(p));
  const auto want = t == 0.5F ? k::ref::warp_blend(pair.a, pair.b, pair.motion)
                              : k::ref::warp_blend_at(pair.a, pair.b, pair.motion, t);
  REQUIRE(want);
  TensorBuffer a = upload_rgb(rig, pair.a, p);
  TensorBuffer b = upload_rgb(rig, pair.b, p);
  TensorBuffer flow = upload_planes(rig, pair.motion.flow, p);
  const std::array mask_plane{pair.motion.mask};
  TensorBuffer mask = upload_planes(rig, mask_plane, p);
  TensorBuffer dst = tensor_target(rig, a.view.width, a.view.height, 3, p);
  const MotionView motion{
      .flow = flow.view, .mask = mask.view, .area = pair.motion.area, .warp = pair.motion.warp};
  record_and_run(
      rig, [&](Recorder& r) { return rig.kernels().warp_blend(r, a.view, b.view, motion, t, dst.view); });
  CHECK(max_abs_diff(download_rgb(rig, dst), as_stored(*want, p)) <= k::tolerance::kWarp);
}

// Garder les poses (warp_blend_poses) : l'instant de chaque pixel suit le mouvement de la paire précédente.
// previous nul : la paire précédente était tenue (C50-06).
void check_poses(Rig& rig, const Pair& pair, const k::ref::Motion* previous, float t, Precision p) {
  INFO("poses, t = " << t << ", précédente " << (previous ? "présente" : "tenue") << ", précision "
                     << static_cast<int>(p));
  const auto want = k::ref::warp_blend_poses(pair.a, pair.b, pair.motion, previous, t);
  REQUIRE(want);
  TensorBuffer a = upload_rgb(rig, pair.a, p);
  TensorBuffer b = upload_rgb(rig, pair.b, p);
  TensorBuffer flow = upload_planes(rig, pair.motion.flow, p);
  const std::array mask_plane{pair.motion.mask};
  TensorBuffer mask = upload_planes(rig, mask_plane, p);
  TensorBuffer before = previous ? upload_planes(rig, previous->flow, p) : tensor_target(rig, 1, 1, 4, p);
  TensorBuffer dst = tensor_target(rig, a.view.width, a.view.height, 3, p);
  const MotionView motion{
      .flow = flow.view, .mask = mask.view, .area = pair.motion.area, .warp = pair.motion.warp};
  const PoseView poses{.keep = true, .previous = previous ? &before.view : nullptr};
  record_and_run(rig, [&](Recorder& r) {
    return rig.kernels().warp_blend(r, a.view, b.view, motion, t, dst.view, poses);
  });
  const float diff = max_abs_diff(download_rgb(rig, dst), as_stored(*want, p));
  CHECK(diff <= k::tolerance::kWarp);
  // Les poses changent bien l'image par rapport au même instant sans elles : le test ne passe pas à vide.
  const auto plain = k::ref::warp_blend_at(pair.a, pair.b, pair.motion, t);
  REQUIRE(plain);
  if (previous) CHECK(max_abs_diff(*want, *plain) > 1e-4F);
}

}  // namespace

TEST_CASE("warp : garder les poses, avec et sans paire précédente, contre la référence",
          "[vk_compute][warp][poses]") {
  Rig rig = make_rig();
  const Pair pair =
      synthetic_pair(80, 60, 64, 64, {.x = 4, .y = 6, .width = 40, .height = 30}, Precision::F32);
  const k::ref::Motion previous =
      synthetic_motion(64, 64, {.x = 4, .y = 6, .width = 40, .height = 30}, 4.0F, Precision::F32);
  for (const float t : {0.25F, 0.75F}) {
    check_poses(rig, pair, &previous, t, Precision::F32);
    check_poses(rig, pair, nullptr, t, Precision::F32);
  }
}

TEST_CASE("warp : garder les poses en float16", "[vk_compute][warp][poses]") {
  Rig rig = make_rig();
  require_f16(rig);
  const Pair pair =
      synthetic_pair(80, 60, 64, 64, {.x = 4, .y = 6, .width = 40, .height = 30}, Precision::F16);
  const k::ref::Motion previous =
      synthetic_motion(64, 64, {.x = 4, .y = 6, .width = 40, .height = 30}, 4.0F, Precision::F16);
  check_poses(rig, pair, &previous, 0.25F, Precision::F16);
  check_poses(rig, pair, nullptr, 0.75F, Precision::F16);
}

TEST_CASE("warp : mouvement estimé à t = 1/2 et autres instants, en float", "[vk_compute][warp]") {
  Rig rig = make_rig();
  // Image 80 × 60, mouvement dans la fenêtre 40 × 30 de plans 64 × 64 : sx = sy = 2.
  const Pair pair =
      synthetic_pair(80, 60, 64, 64, {.x = 4, .y = 6, .width = 40, .height = 30}, Precision::F32);
  for (const float t : {0.5F, 0.25F, 0.7F}) check_warp(rig, pair, t, Precision::F32);
  // Mouvement à la taille de l'image, sans fenêtre.
  const Pair same =
      synthetic_pair(37, 21, 37, 21, {.x = 0, .y = 0, .width = 37, .height = 21}, Precision::F32);
  check_warp(rig, same, 0.5F, Precision::F32);
  check_warp(rig, same, 0.1F, Precision::F32);
}

TEST_CASE("warp : images lues en Catmull-Rom (warp_blend, CUBIC), en float et en float16",
          "[vk_compute][warp]") {
  Rig rig = make_rig();
  Pair pair = synthetic_pair(80, 60, 64, 64, {.x = 4, .y = 6, .width = 40, .height = 30}, Precision::F32);
  pair.motion.warp = k::RifeWarp::CatmullRom;
  for (const float t : {0.5F, 0.3F}) check_warp(rig, pair, t, Precision::F32);
  // Catmull-Rom diffère bien du bilinéaire sur ces images.
  Pair bilinear = pair;
  bilinear.motion.warp = k::RifeWarp::Bilinear;
  const auto cubic = k::ref::warp_blend(pair.a, pair.b, pair.motion);
  const auto plain = k::ref::warp_blend(bilinear.a, bilinear.b, bilinear.motion);
  REQUIRE(cubic);
  REQUIRE(plain);
  CHECK(max_abs_diff(*cubic, *plain) > 1e-3F);
  if (!rig.context->caps().enabled.storage_buffer_16bit) return;
  Pair half = synthetic_pair(80, 60, 64, 64, {.x = 4, .y = 6, .width = 40, .height = 30}, Precision::F16);
  half.motion.warp = k::RifeWarp::CatmullRom;
  check_warp(rig, half, 0.5F, Precision::F16);
}

TEST_CASE("warp : tenseurs float16", "[vk_compute][warp]") {
  Rig rig = make_rig();
  require_f16(rig);
  const Pair pair =
      synthetic_pair(80, 60, 64, 64, {.x = 4, .y = 6, .width = 40, .height = 30}, Precision::F16);
  check_warp(rig, pair, 0.5F, Precision::F16);
  check_warp(rig, pair, 0.3F, Precision::F16);
}

TEST_CASE("warp : instants, fenêtres et tenseurs hors contrat refusés", "[vk_compute][warp]") {
  Rig rig = make_rig();
  KernelRunner& kr = rig.kernels();
  TensorBuffer a = tensor_target(rig, 32, 16, 3, Precision::F32);
  TensorBuffer b = tensor_target(rig, 32, 16, 3, Precision::F32);
  TensorBuffer small = tensor_target(rig, 30, 16, 3, Precision::F32);
  TensorBuffer flow = tensor_target(rig, 16, 8, 4, Precision::F32);
  TensorBuffer mask = tensor_target(rig, 16, 8, 1, Precision::F32);
  TensorBuffer dst = tensor_target(rig, 32, 16, 3, Precision::F32);
  const MotionView motion{
      .flow = flow.view, .mask = mask.view, .area = {.x = 0, .y = 0, .width = 16, .height = 8}};
  Recorder r = require_value(Recorder::begin(*rig.context, *rig.memory));
  for (const float t : {0.0F, 1.0F, -0.5F, std::numeric_limits<float>::quiet_NaN()}) {
    require_refusal(kr.warp_blend(r, a.view, b.view, motion, t, dst.view), RefusalCode::FieldOutOfRange);
  }
  require_refusal(kr.warp_blend(r, a.view, small.view, motion, 0.5F, dst.view), RefusalCode::InputMismatch);
  for (const k::Window area : {k::Window{.x = 1, .y = 0, .width = 16, .height = 8},
                               k::Window{.x = 0, .y = 0, .width = 0, .height = 8},
                               k::Window{.x = -1, .y = 0, .width = 4, .height = 4},
                               k::Window{.x = 0, .y = 4, .width = 4, .height = 5}}) {
    MotionView outside = motion;
    outside.area = area;
    require_refusal(kr.warp_blend(r, a.view, b.view, outside, 0.5F, dst.view), RefusalCode::InputMismatch);
  }
  MotionView three_planes = motion;
  three_planes.flow.planes = 3;
  require_refusal(kr.warp_blend(r, a.view, b.view, three_planes, 0.5F, dst.view), RefusalCode::InputMismatch);
  if (rig.context->caps().enabled.storage_buffer_16bit) {
    TensorBuffer half_mask = tensor_target(rig, 16, 8, 1, Precision::F16);
    MotionView mixed = motion;
    mixed.mask = half_mask.view;
    require_refusal(kr.warp_blend(r, a.view, b.view, mixed, 0.5F, dst.view), RefusalCode::InputMismatch);
  }
  CHECK(require_value(r.finish())->dispatches() == 0);
}
