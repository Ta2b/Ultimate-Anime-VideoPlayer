// Mouvement d'ensemble du warp sous Vulkan (pan_global.comp, warp_blend.glsl) contre kernels/ref.
#include <catch2/catch_test_macros.hpp>

#include "../kernels/warp_pan_test.h"
#include "backend_rig.h"
#include "kernels/ref/warp.h"
#include "kernels/spec/tolerance.h"

using namespace ajp;
using namespace ajp::gpu;
using namespace ajp::gpu::vk::test;
namespace k = ajp::kernels;

TEST_CASE("KernelOps : warp avec le mouvement d'ensemble, contre la référence",
          "[vk_backend][kernels][pan]") {
  TempDir dir("pan");
  auto backend = make_backend(dir.path());
  KernelOps& ops = *backend->kernels();
  constexpr int w = 64;
  constexpr int h = 48;
  for (const bool blind : {true, false}) {
    // Flot nul (RIFE aveugle) ou déjà égal au mouvement d'ensemble : le même résultat attendu.
    INFO("flot " << (blind ? "nul" : "exact"));
    const TensorType type = TensorType::F32;
    const k::RgbImage a = stored(test::pan_image(w, h, 0.0F, 0.0F), type);
    const k::RgbImage b = stored(test::pan_image(w, h, test::kPanDx, test::kPanDy), type);
    k::ref::Motion motion = blind ? test::pan_motion(w, h, 0.0F, 0.0F, 0.0F, 0.0F)
                                  : test::pan_motion(w, h, -test::kPanDx / 2, -test::kPanDy / 2,
                                                     test::kPanDx / 2, test::kPanDy / 2);
    motion.pan_lock = true;
    auto ta = rgb_tensor(*backend, w, h, type);
    auto tb = rgb_tensor(*backend, w, h, type);
    auto flow = rgb_tensor(*backend, w, h, type, 4);
    auto mask = rgb_tensor(*backend, w, h, type, 1);
    auto out = rgb_tensor(*backend, w, h, type);
    write_tensor(*backend, ta->tensor(), rgb_bytes(a, type));
    write_tensor(*backend, tb->tensor(), rgb_bytes(b, type));
    write_tensor(*backend, flow->tensor(), planes_bytes(motion.flow, type));
    write_tensor(*backend, mask->tensor(), planes_bytes(std::span(&motion.mask, 1), type));
    // at_instant : le flot est celui de t (DistilDRBA, RIFE direct), le mouvement d'ensemble suit t.
    for (const bool at : {false, true}) {
      motion.at_instant = at;
      const Motion m{.flow = flow->tensor(),
                     .mask = mask->tensor(),
                     .area = motion.area,
                     .pan_lock = true,
                     .at_instant = at};
      for (const float t : {0.5F, 0.25F, 0.75F}) {
        CAPTURE(at, t);
        REQUIRE(ops.warp_blend(ta->tensor(), tb->tensor(), m, t, out->tensor()));
        const auto want =
            t == 0.5F ? k::ref::warp_blend(a, b, motion) : k::ref::warp_blend_at(a, b, motion, t);
        REQUIRE(want);
        const std::size_t element = 4;
        const k::RgbImage got = rgb_from(
            read_tensor(*backend, out->tensor(), element * 3 * static_cast<std::size_t>(w * h)), w, h, type);
        CHECK(max_abs_diff(got, *want) <= k::tolerance::kWarp);
      }
    }
  }
}
