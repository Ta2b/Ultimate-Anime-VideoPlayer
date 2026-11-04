// Le profil GPU (gpu/gpu_profile.h, chantier 0.2) : médiane, 95e centile et total.
#include "gpu/gpu_profile.h"

#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using ajp::gpu::GpuStageSummary;
using Catch::Approx;

TEST_CASE("profil GPU : médiane, p95 et total", "[gpu][profile]") {
  std::vector<float> samples;
  for (int i = 100; i >= 1; --i) samples.push_back(static_cast<float>(i));  // dans le désordre
  const GpuStageSummary s = ajp::gpu::summarize("moteur", samples);
  CHECK(s.name == "moteur");
  CHECK(s.count == 100);
  CHECK(s.total_ms == Approx(5050.0));
  CHECK(s.median_ms == Approx(51.0));
  CHECK(s.p95_ms == Approx(96.0));
  const GpuStageSummary empty = ajp::gpu::summarize("rien", {});
  CHECK(empty.count == 0);
  CHECK(empty.total_ms == 0.0);
  const std::vector<GpuStageSummary> stages{s, empty};
  const std::string tsv = ajp::gpu::gpu_profile_tsv(stages);
  CHECK(tsv.starts_with(ajp::gpu::kGpuProfileHeader));
  CHECK(tsv.find("moteur\t100\t5050.000\t51.000\t96.000\n") != std::string::npos);
}
