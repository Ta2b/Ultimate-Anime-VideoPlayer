// Le profil GPU (gpu/gpu_profile.h, chantier 0.2) : médiane, 95e centile et total.
#include "gpu/gpu_profile.h"

#include <algorithm>
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

TEST_CASE("profil GPU : passé le plafond, compte et total exacts, centiles de toute la séance (A03-22)",
          "[gpu][profile]") {
  ajp::gpu::StageSamples samples(1000);
  // 10 000 mesures à 1 ms, puis 10 000 à 9 ms : garder les premières seules donnerait une médiane de 1 ms
  // et un 95e centile de 1 ms, comme si la seconde moitié de la séance n'avait pas eu lieu.
  for (int i = 0; i < 10'000; ++i) samples.add(1.0F);
  CHECK(samples.sampled());
  for (int i = 0; i < 10'000; ++i) samples.add(9.0F);
  CHECK(samples.sampled());
  CHECK(samples.count() == 20'000);
  CHECK(samples.total_ms() == Approx(100'000.0));
  CHECK(samples.min_ms() == 1.0F);
  REQUIRE(samples.samples().size() == 1000);
  const auto late = std::ranges::count(samples.samples(), 9.0F);
  CHECK(late > 400);  // la moitié des mesures, à l'aléa du tirage près
  CHECK(late < 600);
  const GpuStageSummary s = samples.summary("étape");
  CHECK(s.count == 20'000);
  CHECK(s.total_ms == Approx(100'000.0));
  CHECK(s.p95_ms == Approx(9.0));

  // Sous le plafond : toutes les mesures, rien de tiré.
  ajp::gpu::StageSamples few(8);
  for (const float ms : {3.0F, 1.0F, 2.0F}) few.add(ms);
  CHECK_FALSE(few.sampled());
  CHECK(few.summary("peu").median_ms == Approx(2.0));
  CHECK(few.min_ms() == 1.0F);
}
