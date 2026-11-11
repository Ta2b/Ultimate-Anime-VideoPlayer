// Les bords de la pré-analyse des plans (chantier D06 de l'audit 2) : la part d'un fondu où la lumière
// change, l'origine d'un flux vidéo qui commence tard, le cache abîmé ou laissé en route, la moyenne
// glissante des scènes vues à l'ingestion.
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

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

// Une image uniforme de luminance moyenne luma (histogramme étalé sur neuf cases autour d'elle).
ShotFrame flat(std::int64_t index, float luma) {
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
  f.grid.fill(luma);
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

TEST_CASE("plans : un plan fini par un fondu reste stable jusqu'au fondu (B42-03)", "[media][shots]") {
  // Une seconde stable à 0,5, puis une seconde et demie qui tombe au noir, sans coupe.
  std::vector<ShotFrame> frames;
  for (int i = 0; i < 24; ++i) frames.push_back(flat(i, 0.5F));
  for (int i = 1; i <= 36; ++i)
    frames.push_back(flat(23 + i, 0.5F * (1.0F - (static_cast<float>(i) / 36.0F))));
  const std::vector<Shot> shots = cut(frames);
  REQUIRE(shots.size() == 1);
  REQUIRE(shots.at(0).fade);
  CHECK(shots.at(0).ramp_to_us > shots.at(0).ramp_from_us);
  ShotTimeline timeline;
  timeline.add(shots.at(0));
  // Au milieu de la partie stable, la lumière est celle du début : elle ne dérivait plus vers le noir.
  REQUIRE(timeline.at(12 * kFrame));
  CHECK(timeline.at(12 * kFrame)->stats.mean == Approx(0.5F).margin(0.01));
  // Le fondu est suivi : à la fin, la lumière est celle de la fin du plan.
  CHECK(timeline.at(59 * kFrame)->stats.mean < 0.2F);
}

TEST_CASE("plans : un fondu qui couvre tout le plan reste interpolé d'un bout à l'autre", "[media][shots]") {
  std::vector<ShotFrame> frames;
  for (int i = 0; i < 72; ++i) frames.push_back(flat(i, 0.6F * (1.0F - static_cast<float>(i) / 72.0F)));
  const std::vector<Shot> shots = cut(frames);
  REQUIRE(shots.size() == 1);
  REQUIRE(shots.at(0).fade);
  ShotTimeline timeline;
  timeline.add(shots.at(0));
  const float begin = timeline.at(0)->stats.mean;
  const float middle = timeline.at(36 * kFrame)->stats.mean;
  const float end = timeline.at(71 * kFrame)->stats.mean;
  CHECK(begin > middle);
  CHECK(middle > end);
}

TEST_CASE("plans : un flux vidéo qui commence tard ne laisse pas de trou en tête (B42-04)",
          "[media][shots]") {
  ShotTimeline timeline;
  timeline.add(Shot{.start_us = 400'000, .end_us = 5'000'000});
  timeline.add(Shot{.start_us = 5'000'000, .end_us = 9'000'000});
  CHECK(timeline.first_gap_us() == 0);  // l'origine inconnue : le début manque
  timeline.set_origin_us(400'000);      // la première image vue par une analyse partie du début
  CHECK(timeline.first_gap_us() == timeline.end_us());
}

TEST_CASE("plans : un cache tronqué au compte énorme est refusé (B42-06)", "[media][shots]") {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "ajp-shot-cache-edges";
  std::filesystem::remove_all(dir);
  const std::string key = "cccccccccccccccc";
  std::vector<Shot> shots(1);
  shots.at(0) = Shot{.start_us = 0, .end_us = 1'000'000};
  REQUIRE(save_shot_cache(dir, key, shots, true));
  REQUIRE(load_shot_cache(dir, key));
  const std::filesystem::path file = dir / (key + ".shots");
  std::string bytes;
  {
    std::ifstream in(file, std::ios::binary);
    bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  }
  // Après la magie (8), la version de l'analyse (4) et la variante (4) : le nombre de plans.
  REQUIRE(bytes.size() > 20);
  const std::uint32_t huge = 199'999;
  std::memcpy(bytes.data() + 16, &huge, sizeof huge);
  std::ofstream(file, std::ios::binary | std::ios::trunc) << bytes;
  CHECK_FALSE(load_shot_cache(dir, key));
  std::filesystem::remove_all(dir);
}

TEST_CASE("plans : les restes d'une écriture coupée du cache sont retirés (B33-07, B36-04)",
          "[media][shots]") {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "ajp-shot-cache-orphans";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const auto touch = [&](const std::string& name, std::chrono::hours age) {
    const std::filesystem::path path = dir / name;
    std::ofstream(path, std::ios::binary) << "reste";
    std::filesystem::last_write_time(path, std::filesystem::file_time_type::clock::now() - age);
    return path;
  };
  const auto old_part = touch("dddddddddddddddd.shots.part", std::chrono::hours(2));
  const auto old_tmp = touch(".dddddddddddddddd.shots.123-4.tmp", std::chrono::hours(2));
  const auto fresh = touch("eeeeeeeeeeeeeeee.shots.part", std::chrono::hours(0));
  std::vector<Shot> shots(1);
  shots.at(0) = Shot{.start_us = 0, .end_us = 1'000'000};
  REQUIRE(save_shot_cache(dir, "ffffffffffffffff", shots, true));
  CHECK_FALSE(std::filesystem::exists(old_part));
  CHECK_FALSE(std::filesystem::exists(old_tmp));
  CHECK(std::filesystem::exists(fresh));  // une écriture peut-être en cours
  // Aucun temporaire laissé par l'écriture elle-même.
  for (const auto& entry : std::filesystem::directory_iterator(dir))
    CHECK(entry.path().extension() != ".tmp");
  std::filesystem::remove_all(dir);
}

TEST_CASE("plans : la scène vue à l'ingestion suit une dérive lente de la lumière (B42-12)",
          "[media][shots]") {
  // Un dégradé horizontal autour de v : un histogramme étalé, comme une vraie image (un aplat uniforme
  // changerait de case à chaque pas et ferait une coupe).
  const auto gray = [](float v) {
    ajp::kernels::RgbImage image(64, 36);
    for (int y = 0; y < 36; ++y)
      for (int x = 0; x < 64; ++x)
        for (auto& plane : image.planes) plane.at(x, y) = v * (0.5F + (static_cast<float>(x) / 64.0F));
    const auto counts = ajp::kernels::ref::frame_counts(image, ajp::kernels::FrameTransfer::Sdr);
    REQUIRE(counts);
    return ajp::kernels::ref::frame_stats_of(*counts);
  };
  LiveScenes live;
  std::int64_t index = 0;
  for (; index < 600; ++index) live.push(index * kFrame, gray(0.5F));  // 25 s de plan fixe
  const auto before = live.at((index - 1) * kFrame);
  REQUIRE(before);
  // La lumière baisse lentement, sans coupe : de 0,5 à 0,4 en dix secondes.
  for (int step = 1; step <= 240; ++step, ++index)
    live.push(index * kFrame, gray(0.5F - (0.1F * static_cast<float>(step) / 240.0F)));
  const auto after = live.at((index - 1) * kFrame);
  REQUIRE(after);
  CHECK(after->serial == before->serial);  // la même scène
  // La moyenne cumulée resterait vers 0,486 ; la moyenne glissante suit la baisse.
  CHECK(after->stats.mean < 0.45F);
}

// B30-01 : une analyse suspendue (une lecture bloquée sur un partage réseau muet, ici le cache qui ne rend
// pas la main) ne fige plus la fermeture : passé stop_limit, son thread est confié à abandon. Avant, la
// destruction joignait le thread sans délai, sur le thread de la fenêtre.
TEST_CASE("plans : la fermeture n'attend pas une analyse suspendue (B30-01)", "[media][shots]") {
  auto release = std::make_shared<std::atomic<bool>>(false);
  std::jthread abandoned;
  std::function<bool()> abandoned_stopped;
  ShotProbeOptions options;
  options.preload = [release](ShotTimeline& /*timeline*/) {
    while (!release->load()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return true;
  };
  options.abandon = [&](std::jthread thread, std::function<bool()> stopped) {
    abandoned = std::move(thread);
    abandoned_stopped = std::move(stopped);
  };
  options.stop_limit = std::chrono::milliseconds(50);
  auto timeline = std::make_shared<ShotTimeline>();
  auto probe = ShotProbe::start("inexistant.mkv", timeline, 0, options);
  const auto before = std::chrono::steady_clock::now();
  probe.reset();
  CHECK(std::chrono::steady_clock::now() - before < std::chrono::seconds(2));
  REQUIRE(abandoned.joinable());
  CHECK_FALSE(abandoned_stopped());
  release->store(true);  // la lecture rend la main : le thread finit seul, sur l'état qu'il partage
  abandoned.join();
  CHECK(abandoned_stopped());
}
