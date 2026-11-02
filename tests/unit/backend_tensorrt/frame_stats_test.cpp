// L'analyse d'une image pour le grade sous TensorRT (stats_kernels.cu) contre la référence
// kernels/ref/frame_stats.cpp, comme le noyau Vulkan (slice_ops_test.cpp). Sans GPU NVIDIA : SKIP.
#include "kernels/ref/frame_stats.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <optional>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "backend_fixture.h"
#include "gpu/backend.h"
#include "kernels/ref/half.h"

namespace {

using namespace ajp;
using namespace ajp::test;
namespace k = ajp::kernels;

}  // namespace

TEST_CASE("Backend TensorRT : analyse d'une image pour le grade, contre la référence",
          "[backend_tensorrt][cuda]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  gpu::KernelOps& ops = *loaded.backend->kernels();
  constexpr int w = 200, h = 120;
  // Un dégradé coloré, des bandes noires de 12 lignes, un teint et quelques points très clairs.
  k::RgbImage image(w, h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const bool bar = y < 12 || y >= h - 12;
      const bool skin = !bar && x < 60 && y < 50;
      const bool spark = !bar && x % 37 == 5 && y % 23 == 7;
      std::array<float, 3> v{static_cast<float>(x) / w, static_cast<float>(y) / h,
                             0.5F + 0.4F * std::sin(0.1F * static_cast<float>(x + y))};
      if (skin) v = {0.85F, 0.66F, 0.5F};
      if (spark) v = {1.0F, 1.0F, 0.95F};
      if (bar) v = {0.0F, 0.0F, 0.0F};
      for (std::size_t c = 0; c < 3; ++c) {
        // Arrondie en FP16 comme le tenseur la garde.
        image.planes[c].at(x, y) = k::ref::half_to_float(k::ref::float_to_half(v[c]));
      }
    }
  }
  auto tensor = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, h, w}});
  REQUIRE(tensor);
  std::vector<float> planes;
  for (const auto& plane : image.planes) {
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) planes.push_back(plane.at(x, y));
  }
  upload((*tensor)->tensor(), planes);
  const float point = 1.0F / (k::frame_stats::kGridW * k::frame_stats::kGridH);
  for (const k::FrameTransfer transfer :
       {k::FrameTransfer::Sdr, k::FrameTransfer::Pq, k::FrameTransfer::Hlg}) {
    INFO("transfert " << static_cast<int>(transfer));
    const auto counts = k::ref::frame_counts(image, transfer);
    REQUIRE(counts);
    const k::FrameStats want = k::ref::frame_stats_of(*counts);
    const auto ticket = ops.frame_stats_start((*tensor)->tensor(), transfer);
    REQUIRE(ticket);
    std::optional<k::FrameStats> got;
    for (int tries = 0; tries < 2000 && !got; ++tries) {
      auto taken = ops.frame_stats_take(*ticket);
      REQUIRE(taken);
      got = *taken;
      if (!got) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(got);
    CHECK_FALSE(ops.frame_stats_take(*ticket));  // relu une fois
    CHECK(got->top == want.top);
    CHECK(got->bottom == want.bottom);
    CHECK(got->left == want.left);
    CHECK(got->right == want.right);
    CHECK(std::abs(got->skin - want.skin) <= 2.0F * point);
    CHECK(std::abs(got->bright - want.bright) <= 4.0F * point);
    CHECK(std::abs(got->highlights - want.highlights) <= 4.0F * point);
    CHECK(std::abs(got->mean - want.mean) <= 1e-4F);
    CHECK(std::abs(got->chroma - want.chroma) <= 1e-4F);
    CHECK(std::abs(got->apl_nits - want.apl_nits) <= 1e-3F * std::max(want.apl_nits, 1.0F));
    float maxrgb = 0.0F, ictcp = 0.0F, luma = 0.0F;
    for (std::size_t i = 0; i < want.maxrgb_hist.size(); ++i) {
      maxrgb += std::abs(got->maxrgb_hist[i] - want.maxrgb_hist[i]);
      ictcp += std::abs(got->i_hist[i] - want.i_hist[i]);
    }
    for (std::size_t i = 0; i < want.luma_hist.size(); ++i)
      luma += std::abs(got->luma_hist[i] - want.luma_hist[i]);
    CHECK(maxrgb <= 8.0F * point);
    CHECK(ictcp <= 8.0F * point);
    CHECK(luma <= 8.0F * point);
  }
}
