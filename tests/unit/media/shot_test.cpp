// La pré-analyse des plans : centiles, découpe (coupes, éclairs, fondus), chronologie, mesure d'une image et
// analyse de la vidéo de calibration livrée (huit plans fixes de 6 s).
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <thread>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "media/shot_cutter.h"
#include "media/shot_probe.h"
#include "media/shot_timeline.h"

using namespace ajp::media;
using Catch::Approx;

namespace {

constexpr std::int64_t kFrame = 41'667;  // 24 im/s

// Une image de luminance moyenne luma, ses points étalés sur neuf cases autour d'elle comme dans un vrai
// plan, avec un motif de grille propre au « cadrage ».
ShotFrame flat(std::int64_t index, float luma, int framing = 0) {
  ShotFrame f;
  f.pts_us = index * kFrame;
  const float center = luma * kShotBins;
  float total = 0.0F;
  for (int b = 0; b < kShotBins; ++b) {
    const float w = std::max(0.0F, 5.0F - std::abs(static_cast<float>(b) + 0.5F - center));
    f.histogram.at(static_cast<std::size_t>(b)) = w;
    total += w;
  }
  for (float& part : f.histogram) part /= total;
  for (std::size_t i = 0; i < f.grid.size(); ++i)
    f.grid.at(i) = luma + (framing != 0 && (i + static_cast<std::size_t>(framing)) % 2 == 0 ? 0.3F : 0.0F);
  f.mean = luma;
  return f;
}

std::vector<Shot> cut(const std::vector<ShotFrame>& frames) {
  ShotCutter cutter;
  std::vector<Shot> shots;
  for (const ShotFrame& f : frames)
    if (auto s = cutter.push(f)) shots.push_back(*s);
  if (auto s = cutter.finish(frames.back().pts_us + kFrame)) shots.push_back(*s);
  return shots;
}

}  // namespace

TEST_CASE("plans : centiles et parts d'un histogramme", "[media][shots]") {
  std::array<float, kShotBins> uniform{};
  uniform.fill(1.0F / kShotBins);
  const ShotStats s = stats_of(uniform, 0.5F, 0.0F, 0.0F);
  CHECK(s.p05 == Approx(0.05F).margin(0.002));
  CHECK(s.p50 == Approx(0.50F).margin(0.002));
  CHECK(s.p95 == Approx(0.95F).margin(0.002));
  CHECK(s.dark == Approx(0.10F).margin(0.002));
  CHECK(s.bright == Approx(0.20F).margin(0.002));
}

TEST_CASE("plans : une coupe franche sépare deux plans, à l'image près", "[media][shots]") {
  std::vector<ShotFrame> frames;
  for (int i = 0; i < 48; ++i) frames.push_back(flat(i, 0.12F));
  for (int i = 48; i < 96; ++i) frames.push_back(flat(i, 0.65F, 1));
  const std::vector<Shot> shots = cut(frames);
  REQUIRE(shots.size() == 2);
  CHECK(shots.at(0).start_us == 0);
  CHECK(shots.at(0).end_us == 48 * kFrame);
  CHECK(shots.at(1).start_us == 48 * kFrame);
  CHECK(shots.at(1).end_us == 96 * kFrame);
  CHECK_FALSE(shots.at(0).fade);
  CHECK(shots.at(0).first.mean == Approx(0.12F));
  CHECK(shots.at(1).first.mean == Approx(0.65F));
}

TEST_CASE("plans : deux plans de même luminosité, cadrés autrement, sont coupés", "[media][shots]") {
  std::vector<ShotFrame> frames;
  for (int i = 0; i < 40; ++i) frames.push_back(flat(i, 0.4F, 0));
  for (int i = 40; i < 80; ++i) frames.push_back(flat(i, 0.4F, 1));
  REQUIRE(cut(frames).size() == 2);
}

TEST_CASE("plans : un éclair d'une image n'est pas une coupe", "[media][shots]") {
  std::vector<ShotFrame> frames;
  for (int i = 0; i < 60; ++i) frames.push_back(flat(i, i == 30 ? 0.95F : 0.15F));
  REQUIRE(cut(frames).size() == 1);
}

TEST_CASE("plans : un fondu au noir reste un plan, suivi du début à la fin", "[media][shots]") {
  std::vector<ShotFrame> frames;
  for (int i = 0; i < 72; ++i) frames.push_back(flat(i, 0.6F * (1.0F - static_cast<float>(i) / 72.0F)));
  const std::vector<Shot> shots = cut(frames);
  REQUIRE(shots.size() == 1);
  CHECK(shots.at(0).fade);
  CHECK(shots.at(0).first.mean > shots.at(0).last.mean + 0.3F);
}

TEST_CASE("plans : la chronologie rend le plan d'un instant, et interpole un fondu", "[media][shots]") {
  ShotTimeline timeline;
  Shot a{.start_us = 0, .end_us = 1'000'000};
  a.first.mean = 0.2F;
  a.last = a.first;
  Shot b{.start_us = 1'000'000, .end_us = 3'000'000, .fade = true};
  b.first.mean = 0.6F;
  b.last.mean = 0.0F;
  timeline.add(b);
  timeline.add(a);
  REQUIRE(timeline.at(500'000));
  CHECK(timeline.at(500'000)->stats.mean == Approx(0.2F));
  REQUIRE(timeline.at(2'000'000));
  CHECK(timeline.at(2'000'000)->stats.mean == Approx(0.3F));
  CHECK(timeline.at(999'999)->serial != timeline.at(1'000'000)->serial);
  CHECK_FALSE(timeline.at(3'000'000));
  CHECK(timeline.analyzed_until(0) == 3'000'000);
  CHECK(*timeline.mean_luma() == Approx((0.2 * 1.0 + 0.3 * 2.0) / 3.0));
  // Un plan coupé par un départ en cours de vidéo ne remplace pas un plan entier.
  Shot partial{.start_us = 500'000, .end_us = 900'000, .cut_before = false};
  timeline.add(partial);
  CHECK(timeline.at(600'000)->start_us == 0);
}

TEST_CASE("plans : la vidéo de calibration se découpe en ses huit plans de 6 s", "[media][shots]") {
  const std::filesystem::path clip = std::filesystem::path(AJP_DATA_DIR) / "calibration" / "calibration.mkv";
  REQUIRE(std::filesystem::exists(clip));
  ShotTimeline timeline;
  const auto probe = ShotProbe::start(clip, timeline, 0);
  for (int i = 0; i < 600 && !probe->finished(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  REQUIRE(probe->finished());
  REQUIRE_FALSE(probe->failure());
  REQUIRE(timeline.size() == 8);
  for (int i = 0; i < 8; ++i) {
    const auto shot = timeline.at(static_cast<std::int64_t>(i) * 6'000'000 + 3'000'000);
    REQUIRE(shot);
    // Une coupe à 6 s près d'une image (24 im/s).
    CHECK(std::abs(shot->start_us - static_cast<std::int64_t>(i) * 6'000'000) <= 42'000);
    CHECK_FALSE(shot->fade);
  }
  // Les quatre premiers plans sont sombres, les quatre derniers clairs.
  CHECK(timeline.at(3'000'000)->stats.mean < timeline.at(27'000'000)->stats.mean);
}

TEST_CASE("plans : l'analyse repart d'un saut et n'attend pas plus que son avance", "[media][shots]") {
  const std::filesystem::path clip = std::filesystem::path(AJP_DATA_DIR) / "calibration" / "calibration.mkv";
  ShotTimeline timeline;
  const auto probe = ShotProbe::start(clip, timeline, 30'000'000, ShotProbeOptions{.lead_us = 60'000'000});
  for (int i = 0; i < 600 && !probe->finished(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  REQUIRE(probe->finished());
  REQUIRE_FALSE(probe->failure());
  CHECK_FALSE(timeline.at(10'000'000));  // avant le départ : rien
  REQUIRE(timeline.at(40'000'000));
  CHECK(timeline.at(40'000'000)->start_us == Approx(36'000'000).margin(42'000));
}
