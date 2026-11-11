// Le profil GPU (gpu/gpu_profile.h, chantier 0.2) : médiane, 95e centile et total.
#include "gpu/gpu_profile.h"

#include <cmath>
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
  CHECK(s.median_ms == Approx(50.5));  // les deux du milieu, 50 et 51
  CHECK(s.p95_ms == Approx(95.0));     // rang ceil(0,95 × 100) = 95
  const GpuStageSummary empty = ajp::gpu::summarize("rien", {});
  CHECK(empty.count == 0);
  CHECK(empty.total_ms == 0.0);
  const std::vector<GpuStageSummary> stages{s, empty};
  const std::string tsv = ajp::gpu::gpu_profile_tsv(stages);
  CHECK(tsv.starts_with(ajp::gpu::kGpuProfileHeader));
  CHECK(tsv.find("moteur\t100\t5050.000\t50.500\t95.000\n") != std::string::npos);
}

TEST_CASE("profil GPU : centile au rang le plus proche, mesures non finies écartées (B21-09)",
          "[gpu][profile]") {
  // Un petit échantillon : le p95 n'est pas d'office le maximum.
  const GpuStageSummary five = ajp::gpu::summarize("cinq", {1.0F, 2.0F, 3.0F, 4.0F, 100.0F});
  CHECK(five.median_ms == Approx(3.0));
  CHECK(five.p95_ms == Approx(100.0));  // ceil(0,95 × 5) = 5 : ici c'est bien le cinquième
  std::vector<float> forty;
  for (int i = 1; i <= 40; ++i) forty.push_back(static_cast<float>(i));
  const GpuStageSummary s40 = ajp::gpu::summarize("quarante", forty);
  CHECK(s40.p95_ms == Approx(38.0));  // ceil(38) = 38, et pas 39 (40 × 95 / 100 = 38, compté depuis 0)
  CHECK(s40.median_ms == Approx(20.5));
  const GpuStageSummary odd = ajp::gpu::summarize("nan", {2.0F, std::nanf(""), 1.0F, INFINITY, 3.0F});
  CHECK(odd.count == 3);
  CHECK(odd.median_ms == Approx(2.0));
  CHECK(odd.total_ms == Approx(6.0));
}
