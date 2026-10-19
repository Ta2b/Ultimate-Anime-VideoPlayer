// Banc caché ([.bench]) : coût de la composition et du grade par image 4K, sur le GPU de la
// machine. À lancer sous le verrou GPU :
//   ajp-gpu build/linux-prefix/tests/unit/render/ajp_render_tests "[.bench]"
// Temps GPU : somme des passes mesurée par libplacebo ; temps mur : rendu puis pl_gpu_finish.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "render/compositor.h"
#include "support.h"

namespace ajp::render {
namespace {

struct Stats {
  double median = 0.0;
  double stddev = 0.0;
};

Stats stats(std::vector<double> values) {
  std::ranges::sort(values);
  const double mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
  double var = 0.0;
  for (const double v : values) var += (v - mean) * (v - mean);
  return {.median = values[values.size() / 2], .stddev = std::sqrt(var / static_cast<double>(values.size()))};
}

}  // namespace

TEST_CASE("banc : composition et grade d'une image 4K", "[.bench]") {
  std::string reason;
  auto device = test::make_device(reason);
  if (!device) SKIP("pas d'appareil Vulkan : " << reason);
  constexpr int w = 3840;
  constexpr int h = 2160;
  constexpr int warmup = 20;
  constexpr int runs = 200;
  std::vector<std::uint16_t> pixels(static_cast<std::size_t>(w) * h * 4);
  for (std::size_t i = 0; i < pixels.size(); ++i)
    pixels[i] = static_cast<std::uint16_t>((i * 2654435761U) >> 16U);
  pl_tex slot = test::make_texture(device->gpu(), w, h, pixels);
  REQUIRE(slot != nullptr);
  auto target =
      OffscreenTarget::create(*device, {.width = w, .height = h, .display = test::pq_display(1000.0)});
  REQUIRE(target);
  const SlotColor color{.primaries = Primaries::Bt709, .transfer = Transfer::Bt1886};

  for (const bool grade : {false, true}) {
    auto compositor = Compositor::create(*device);
    REQUIRE(compositor);
    REQUIRE(compositor->configure({.display = test::pq_display(1000.0), .route = core::HdrMode::Native}));
    compositor->set_grade_enabled(grade);
    std::vector<double> gpu_ms;
    std::vector<double> wall_ms;
    for (int i = 0; i < warmup + runs; ++i) {
      const auto start = std::chrono::steady_clock::now();
      REQUIRE(compositor->compose(slot, color, target->frame()));
      pl_gpu_finish(device->gpu());
      const auto wall =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
      if (i < warmup) continue;
      wall_ms.push_back(wall);
      gpu_ms.push_back(static_cast<double>(compositor->last_gpu_time_ns()) / 1e6);
    }
    const Stats g = stats(gpu_ms);
    const Stats m = stats(wall_ms);
    std::printf("%s, 4K PQ, %s : GPU %.3f ms (écart type %.3f), mur %.3f ms (écart type %.3f), %d passes\n",
                device->description().name.c_str(), grade ? "grade" : "sans grade", g.median, g.stddev,
                m.median, m.stddev, runs);
  }
  pl_tex_destroy(device->gpu(), &slot);
}

}  // namespace ajp::render
