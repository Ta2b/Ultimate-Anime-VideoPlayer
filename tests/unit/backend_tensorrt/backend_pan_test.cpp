// Mouvement d'ensemble du warp sous TensorRT (pan_global et warp_pixel de warp_kernels.cu) contre
// kernels/ref. Sans GPU NVIDIA, le test se SKIP.
#include <algorithm>
#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "../kernels/warp_pan_test.h"
#include "backend_fixture.h"
#include "gpu/backend.h"
#include "kernels/ref/half.h"
#include "kernels/ref/warp.h"

namespace {

using namespace ajp;
using namespace ajp::test;

}  // namespace

TEST_CASE("Backend TensorRT : warp avec le mouvement d'ensemble, contre la référence",
          "[backend_tensorrt][cuda][pan]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  constexpr int w = 64;
  constexpr int h = 48;
  const auto q = [](float v) { return kernels::ref::half_to_float(kernels::ref::float_to_half(v)); };
  const auto planar = [&](const kernels::RgbImage& image) {
    std::vector<float> v(static_cast<std::size_t>(3 * w * h));
    for (int c = 0; c < 3; ++c) {
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x)
          v[static_cast<std::size_t>((c * h + y) * w + x)] =
              image.planes[static_cast<std::size_t>(c)].at(x, y);
      }
    }
    return v;
  };
  kernels::RgbImage a = pan_image(w, h, 0.0F, 0.0F);
  kernels::RgbImage b = pan_image(w, h, kPanDx, kPanDy);
  for (auto* image : {&a, &b}) {
    for (auto& plane : image->planes) std::ranges::transform(plane.data, plane.data.begin(), q);
  }
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  auto alloc = [&](std::int64_t c) {
    auto t = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, c, h, w}});
    REQUIRE(t);
    return std::move(*t);
  };
  auto ta = alloc(3);
  auto tb = alloc(3);
  auto out = alloc(3);
  auto tflow = alloc(4);
  auto tmask = alloc(1);
  upload(ta->tensor(), planar(a));
  upload(tb->tensor(), planar(b));
  for (const bool blind : {true, false}) {
    INFO("flot " << (blind ? "nul" : "exact"));
    // Flot nul (RIFE aveugle) ou égal au mouvement d'ensemble au milieu ; valeurs exactes en FP16.
    const float hx = blind ? 0.0F : q(kPanDx / 2);
    const float hy = blind ? 0.0F : q(kPanDy / 2);
    kernels::ref::Motion motion = pan_motion(w, h, -hx, -hy, hx, hy);
    motion.pan_lock = true;
    std::vector<float> flow(static_cast<std::size_t>(4 * w * h));
    for (int c = 0; c < 4; ++c)
      std::ranges::fill_n(flow.begin() + static_cast<std::ptrdiff_t>(c * w * h), w * h,
                          motion.flow[static_cast<std::size_t>(c)].at(0, 0));
    upload(tflow->tensor(), flow);
    upload(tmask->tensor(), std::vector<float>(static_cast<std::size_t>(w * h), 0.5F));
    // at_instant : le flot est celui de t (DistilDRBA, RIFE direct), le mouvement d'ensemble suit t.
    for (const bool at : {false, true}) {
      motion.at_instant = at;
      const gpu::Motion gm{.flow = tflow->tensor(),
                           .mask = tmask->tensor(),
                           .area = motion.area,
                           .pan_lock = true,
                           .at_instant = at};
      for (const float t : {0.5F, 0.25F, 0.75F}) {
        CAPTURE(at, t);
        REQUIRE(loaded.backend->kernels()->warp_blend(ta->tensor(), tb->tensor(), gm, t, out->tensor()));
        const auto got = download(out->tensor());
        const auto want = kernels::ref::warp_blend_at(a, b, motion, t);
        REQUIRE(want);
        double worst = 0.0;
        for (int c = 0; c < 3; ++c) {
          for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
              worst = std::max(
                  worst, static_cast<double>(std::fabs(want->planes[static_cast<std::size_t>(c)].at(x, y) -
                                                       got[static_cast<std::size_t>((c * h + y) * w + x)])));
            }
          }
        }
        CHECK(worst < 2e-3);
      }
    }
  }
}
