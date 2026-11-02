// La pré-analyse des plans : centiles, découpe (coupes, éclairs, fondus), chronologie, mesure d'une image et
// analyse de la vidéo de calibration livrée (huit plans fixes de 6 s).
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "clips.h"
#include "kernels/ref/frame_stats.h"
#include "media/live_scenes.h"
#include "media/shot_cache.h"
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

TEST_CASE("plans : dix-huit plans fixes de 6 s se découpent en dix-huit plans", "[media][shots]") {
  const std::filesystem::path clip = test::require_clip(test::shots_clip());
  ShotTimeline timeline;
  const auto probe = ShotProbe::start(clip, timeline, 0);
  for (int i = 0; i < 600 && !probe->finished(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  REQUIRE(probe->finished());
  REQUIRE_FALSE(probe->failure());
  REQUIRE(timeline.size() == 18);
  for (int i = 0; i < 18; ++i) {
    const auto shot = timeline.at(static_cast<std::int64_t>(i) * 6'000'000 + 3'000'000);
    REQUIRE(shot);
    // Une coupe à 6 s près d'une image (24 im/s).
    CHECK(std::abs(shot->start_us - static_cast<std::int64_t>(i) * 6'000'000) <= 42'000);
    CHECK_FALSE(shot->fade);
  }
  // Le premier plan est une nuit, le cinquième une scène claire.
  CHECK(timeline.at(3'000'000)->stats.mean < timeline.at(27'000'000)->stats.mean);
}

TEST_CASE("plans : l'analyse partie d'un saut revient combler le début", "[media][shots]") {
  const std::filesystem::path clip = test::require_clip(test::shots_clip());
  ShotTimeline timeline;
  const auto probe = ShotProbe::start(clip, timeline, 30'000'000, ShotProbeOptions{.lead_us = 120'000'000});
  for (int i = 0; i < 600 && !probe->finished(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  REQUIRE(probe->finished());
  REQUIRE_FALSE(probe->failure());
  REQUIRE(timeline.at(40'000'000));
  CHECK(timeline.at(40'000'000)->start_us == Approx(36'000'000).margin(42'000));
  // Le début, sauté au départ, est analysé une fois la fin atteinte : la vidéo est connue entière.
  REQUIRE(timeline.at(10'000'000));
  CHECK(timeline.at(10'000'000)->start_us == Approx(6'000'000).margin(42'000));
  CHECK(timeline.first_gap_us() == timeline.end_us());
}

TEST_CASE("plans : le cache rend les plans gardés, et rien pour une autre vidéo", "[media][shots]") {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "ajp-shot-cache-test";
  std::filesystem::remove_all(dir);
  const std::filesystem::path clip = test::require_clip(test::shots_clip());
  const std::string key = shot_cache_key(clip);
  REQUIRE(key.size() == 16);
  CHECK(key == shot_cache_key(clip));
  std::vector<Shot> shots(2);
  shots.at(0) = Shot{.start_us = 0, .end_us = 1'000'000};
  shots.at(0).first.p995 = 0.9F;
  shots.at(1) = Shot{.start_us = 1'000'000, .end_us = 3'000'000, .fade = true};
  shots.at(1).last.mean = 0.4F;
  shots.at(1).meta.max_scl_nits = 812.5F;
  shots.at(1).meta.mean.skin = 0.25F;
  shots.at(1).meta.mean.maxrgb_nits.back() = 640.0F;
  REQUIRE(save_shot_cache(dir, key, shots, true));
  const auto loaded = load_shot_cache(dir, key);
  REQUIRE(loaded);
  CHECK(loaded->complete);
  REQUIRE(loaded->shots.size() == 2);
  CHECK(loaded->shots.at(0).first.p995 == 0.9F);
  CHECK(loaded->shots.at(1).fade);
  CHECK(loaded->shots.at(1).last.mean == 0.4F);
  CHECK(loaded->shots.at(1).meta == shots.at(1).meta);
  CHECK(loaded->shots.at(1).end_us == 3'000'000);
  // Une analyse arrêtée en route se garde aussi, marquée incomplète.
  REQUIRE(save_shot_cache(dir, key, shots, false));
  CHECK_FALSE(load_shot_cache(dir, key)->complete);
  CHECK_FALSE(load_shot_cache(dir, "0123456789abcdef"));
  // Au-delà de la limite, les plus anciens partent.
  REQUIRE(save_shot_cache(dir, "aaaaaaaaaaaaaaaa", shots, true, 1));
  CHECK_FALSE(load_shot_cache(dir, key));
  CHECK(load_shot_cache(dir, "aaaaaaaaaaaaaaaa"));
  std::filesystem::remove_all(dir);
}

TEST_CASE("plans : scènes vues à l'ingestion, coupes et sauts", "[media][shots]") {
  const auto gray = [](float v) {
    const auto counts =
        ajp::kernels::ref::frame_counts(ajp::kernels::RgbImage(64, 36, v), ajp::kernels::FrameTransfer::Sdr);
    REQUIRE(counts);
    return ajp::kernels::ref::frame_stats_of(*counts);
  };
  LiveScenes live;
  CHECK_FALSE(live.at(0));
  for (int i = 0; i < 5; ++i) live.push(i * kFrame, gray(0.2F));
  const auto dark = live.at(4 * kFrame + 1000);
  REQUIRE(dark);
  CHECK(dark->stats.mean == Approx(0.2F));
  CHECK(dark->light.apl_nits == Approx(std::pow(0.2F, 2.4F) * 203.0F).epsilon(1e-4));
  for (int i = 5; i < 8; ++i) live.push(i * kFrame, gray(0.8F));
  const auto bright = live.at(7 * kFrame);
  REQUIRE(bright);
  CHECK(bright->serial != dark->serial);  // une coupe
  CHECK(bright->stats.mean == Approx(0.8F));
  CHECK(live.at(2 * kFrame)->serial == dark->serial);  // l'image d'avant la coupe garde sa scène
  CHECK_FALSE(live.at(7 * kFrame + 600'000));          // trop loin de la dernière analysée
  live.push(kFrame, gray(0.8F));                       // un saut en arrière : tout repart
  CHECK_FALSE(live.at(0));
  CHECK(live.at(kFrame)->serial != bright->serial);
}
