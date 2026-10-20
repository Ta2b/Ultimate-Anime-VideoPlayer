// GMFSS Fortuna par le backend TensorRT (gpu/kernel_ops.h) contre kernels/ref : le softmax
// splatting par additions atomiques CUDA. Sans GPU NVIDIA, le test se SKIP.
#include "kernels/ref/gmfss.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "backend_fixture.h"
#include "kernels/ref/half.h"
#include "kernels/spec/gmfss.h"
#include "kernels/spec/tolerance.h"

namespace {

using namespace ajp;
using namespace ajp::test;
namespace k = ajp::kernels;

constexpr int kW = 40;
constexpr int kH = 24;
constexpr int kChannels = 19;  // deux groupes de plans, le second incomplet

// Des plans dans [lo, hi], déjà arrondis en FP16 comme le tenseur les garde.
std::vector<k::PlaneF> planes(int count, int seed, float lo, float hi) {
  std::vector<k::PlaneF> out;
  for (int c = 0; c < count; ++c) {
    k::PlaneF plane(kW, kH);
    for (int y = 0; y < kH; ++y) {
      for (int x = 0; x < kW; ++x) {
        const float u = 0.5F + 0.5F * std::sin(0.37F * static_cast<float>(x + seed * 7 + c * 3) +
                                               0.23F * static_cast<float>(y - seed));
        plane.at(x, y) = k::ref::half_to_float(k::ref::float_to_half(lo + (hi - lo) * u));
      }
    }
    out.push_back(std::move(plane));
  }
  return out;
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
  const std::vector<k::PlaneF> flow = planes(2, 2, -6.0F, 6.0F);
  const std::vector<k::PlaneF> metric = planes(1, 3, -k::kGmfssMetricLimit, k::kGmfssMetricLimit);
  const k::ref::SplatMotion motion{{flow[0], flow[1], metric[0]}};
  auto ts = alloc(kChannels);
  auto tm = alloc(k::kGmfssMotionChannels);
  auto out = alloc(kChannels);
  upload(ts->tensor(), flat(source));
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
    // Seul l'ordre des additions atomiques change les arrondis, puis le FP16 de la sortie.
    CHECK(worst <= k::tolerance::kRgbF16);
  }
  const auto late = ops.soft_splat(ts->tensor(), tm->tensor(), 1.5F, out->tensor());
  REQUIRE_FALSE(late);
  CHECK(late.error().code == RefusalCode::FieldOutOfRange);
}
