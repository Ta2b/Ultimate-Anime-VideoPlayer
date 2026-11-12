// La pré-analyse des plans : centiles, découpe (coupes, éclairs, fondus), chronologie, mesure d'une image et
// analyse de la vidéo de calibration livrée (huit plans fixes de 6 s).
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "clips.h"
#include "kernels/ref/frame_stats.h"
#include "media/flash_guard.h"
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

TEST_CASE("plans : champ-contrechamp de même luminosité et cadrage, couleurs autres (P3.31)",
          "[media][shots]") {
  std::vector<ShotFrame> frames;
  for (int i = 0; i < 80; ++i) {
    ShotFrame f = flat(i, 0.4F);
    f.blue = i < 40 ? 0.02F : 0.15F;  // un fond rouge, puis un ciel bleu
    f.chroma = i < 40 ? 0.30F : 0.25F;
    frames.push_back(f);
  }
  REQUIRE(cut(frames).size() == 2);
}

TEST_CASE("plans : un éclair d'une image n'est pas une coupe", "[media][shots]") {
  std::vector<ShotFrame> frames;
  for (int i = 0; i < 60; ++i) frames.push_back(flat(i, i == 30 ? 0.95F : 0.15F));
  REQUIRE(cut(frames).size() == 1);
}

TEST_CASE("plans : un éclair tenu deux ou trois images n'est pas une coupe (A42-02)", "[media][shots]") {
  for (const int held : {2, 3}) {
    CAPTURE(held);
    std::vector<ShotFrame> frames;
    for (int i = 0; i < 60; ++i) frames.push_back(flat(i, i >= 30 && i < 30 + held ? 0.95F : 0.15F));
    REQUIRE(cut(frames).size() == 1);
  }
  // Quatre images : un plan court, pas un éclair.
  std::vector<ShotFrame> frames;
  for (int i = 0; i < 60; ++i) frames.push_back(flat(i, i >= 30 && i < 34 ? 0.95F : 0.15F));
  const std::vector<Shot> shots = cut(frames);
  REQUIRE(shots.size() == 3);
  CHECK(shots.at(1).start_us == 30 * kFrame);
  CHECK(shots.at(1).end_us == 34 * kFrame);
}

// Un décor qui défile sur la grille : `speed` colonnes par dessin vu, chaque dessin tenu `held` images.
std::vector<ShotFrame> pan(int count, float speed, int held, int first = 0) {
  std::vector<ShotFrame> frames;
  for (int i = 0; i < count; ++i) {
    ShotFrame f = flat(first + i, 0.4F);
    const float position = speed * static_cast<float>(i / held * held);
    for (std::size_t cell = 0; cell < f.grid.size(); ++cell) {
      const float x = (static_cast<float>(cell % kShotGrid) + position) / static_cast<float>(kShotGrid);
      f.grid.at(cell) = 0.25F + 0.3F * std::abs(2.0F * (x - std::floor(x)) - 1.0F);
    }
    frames.push_back(f);
  }
  return frames;
}

TEST_CASE("plans : un panoramique animé en uns, en deux ou en trois reste un plan (A42-01)",
          "[media][shots]") {
  for (const float speed : {0.25F, 0.35F}) {
    for (const int held : {1, 2, 3}) {
      CAPTURE(speed, held);
      CHECK(cut(pan(96, speed, held)).size() == 1);
    }
  }
}

TEST_CASE("plans : une coupe franche dans une animation tenue reste une coupe, à l'image près",
          "[media][shots]") {
  std::vector<ShotFrame> frames = pan(48, 0.25F, 3);
  for (int i = 48; i < 96; ++i) frames.push_back(flat(i, 0.85F, 1));
  const std::vector<Shot> shots = cut(frames);
  REQUIRE(shots.size() == 2);
  CHECK(shots.at(0).end_us == 48 * kFrame);
  CHECK(shots.at(1).start_us == 48 * kFrame);
}

TEST_CASE("plans : un plan fixe oublie le seuil du panoramique d'avant", "[media][shots]") {
  // Deux secondes de plan fixe après un panoramique rapide, puis une coupe modeste (0,3 à 0,34 de distance) :
  // le seuil monté par le panoramique ne la cache plus.
  std::vector<ShotFrame> frames = pan(48, 0.35F, 1);
  const ShotFrame still = frames.back();
  for (int i = 48; i < 96; ++i) {
    ShotFrame f = still;
    f.pts_us = i * kFrame;
    frames.push_back(f);
  }
  for (int i = 96; i < 144; ++i) {
    ShotFrame f = still;
    f.pts_us = i * kFrame;
    for (float& cell : f.grid) cell += 0.13F;
    frames.push_back(f);
  }
  CHECK(frame_distance(frames.at(95), frames.at(96)) < 0.35F);
  REQUIRE(cut(frames).size() == 2);
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
  // Un plan coupé par un départ en cours de vidéo ne remplace pas un plan entier.
  Shot partial{.start_us = 500'000, .end_us = 900'000, .cut_before = false};
  timeline.add(partial);
  CHECK(timeline.at(600'000)->start_us == 0);
}

TEST_CASE("plans : un plan repris après un saut comble le trou au lieu d'être jeté (A42-03)",
          "[media][shots]") {
  // La lecture saute pendant l'analyse : le plan en cours est clos à 210 s, qui n'est pas une coupe. Revenue
  // là, l'analyse repart de l'image clé de 208 s et rend le plan [208, 215), sans coupe connue devant lui.
  ShotTimeline timeline;
  timeline.add(Shot{.start_us = 0, .end_us = 205'000'000});
  timeline.add(Shot{.start_us = 205'000'000, .end_us = 210'000'000});
  timeline.add(Shot{.start_us = 215'000'000, .end_us = 300'000'000});
  REQUIRE(timeline.first_gap_us() == 210'000'000);
  const std::uint64_t before = timeline.version();
  timeline.add(Shot{.start_us = 208'000'000, .end_us = 215'000'000, .cut_before = false});
  CHECK(timeline.first_gap_us() == 300'000'000);
  REQUIRE(timeline.at(212'000'000));
  CHECK(timeline.at(212'000'000)->start_us == 210'000'000);
  CHECK_FALSE(timeline.at(212'000'000)->cut_before);
  CHECK(timeline.at(209'000'000)->start_us == 205'000'000);  // le plan déjà connu reste entier
  CHECK(timeline.version() != before);
}

TEST_CASE("plans : la version de la chronologie suit un remplacement à nombre égal (A42-11)",
          "[media][shots]") {
  ShotTimeline timeline;
  timeline.add(Shot{.start_us = 0, .end_us = 1'000'000});
  timeline.add(Shot{.start_us = 1'200'000, .end_us = 2'000'000, .cut_before = false});
  const std::uint64_t before = timeline.version();
  timeline.add(Shot{.start_us = 1'000'000, .end_us = 2'000'000});  // le plan entier remplace le partiel
  CHECK(timeline.size() == 2);
  CHECK(timeline.version() != before);
  CHECK(timeline.at(1'500'000)->cut_before);
}

TEST_CASE("plans : les séquences de flashs de la chronologie se fusionnent", "[media][shots]") {
  ShotTimeline timeline;
  CHECK_FALSE(timeline.flashing(500'000));
  timeline.add_flash({.from_us = 1'000'000, .until_us = 2'000'000});
  timeline.add_flash({.from_us = 5'000'000, .until_us = 6'000'000});
  timeline.add_flash({.from_us = 1'500'000, .until_us = 2'500'000});  // la même séquence, qui se prolonge
  CHECK(timeline.flashes() == std::vector<FlashSpan>{{1'000'000, 2'500'000}, {5'000'000, 6'000'000}});
  CHECK(timeline.flashing(1'000'000));
  CHECK(timeline.flashing(2'400'000));
  CHECK_FALSE(timeline.flashing(2'500'000));
  CHECK_FALSE(timeline.flashing(900'000));
  timeline.add_flash({.from_us = 3'000'000, .until_us = 3'000'000});  // vide : ignorée
  CHECK(timeline.flashes().size() == 2);
}

TEST_CASE("plans : la pré-analyse marque une séquence de flashs dès sa première transition (A42-17)",
          "[media][shots]") {
  // Ce que fait ShotProbe image par image : le limiteur sur les zones de l'image, ses séquences dans la
  // chronologie. Deux secondes calmes, une seconde d'alternance à 6 Hz, puis le calme.
  ShotTimeline timeline;
  FlashGuard guard;
  for (int i = 0; i < 120; ++i) {
    const bool storm = i >= 48 && i < 72;
    ShotFrame frame = flat(i, 0.3F);
    frame.light.apl_nits = storm && (i / 2) % 2 == 1 ? 180.0F : 30.0F;
    if (guard.push(frame.pts_us, zone_nits_of(frame)))
      if (const auto from = guard.danger_from())
        timeline.add_flash({.from_us = *from, .until_us = guard.limited_until()});
  }
  REQUIRE(timeline.flashes().size() == 1);
  CHECK(timeline.flashes().front().from_us <= 51 * kFrame);  // la première transition, pas la septième
  CHECK(timeline.flashing(60 * kFrame));
  CHECK_FALSE(timeline.flashing(40 * kFrame));
  CHECK_FALSE(timeline.flashing(110 * kFrame));  // une seconde après la dernière transition, le plafond part
}

TEST_CASE("plans : les zones d'une image en cd/m² gardent sa luminance moyenne", "[media][shots]") {
  ShotFrame frame = flat(0, 0.5F);
  frame.light.apl_nits = 40.0F;
  for (std::size_t i = 0; i < frame.grid.size(); ++i) frame.grid.at(i) = i < 4 ? 0.8F : 0.2F;
  const auto zones = zone_nits_of(frame);
  float sum = 0.0F;
  for (const float z : zones) sum += z;
  CHECK(sum / static_cast<float>(zones.size()) == Approx(40.0F));
  CHECK(zones.at(0) > 10.0F * zones.at(15));  // le quart clair porte presque toute la lumière
}

// L'attente d'une pré-analyse entière, en pas de 100 ms : 5 min.
// D'ordinaire, elle finit en quelques secondes. Sous ASan, avec le lecteur
// entier en CI (ctest -j 3), elle dépassait la minute d'avant. La boucle
// sort dès la fin : rien n'est plus lent quand tout va bien.
constexpr int kProbeWait = 3000;

TEST_CASE("plans : dix-huit plans fixes de 6 s se découpent en dix-huit plans", "[media][shots]") {
  const std::filesystem::path clip = test::require_clip(test::shots_clip());
  const auto shared = std::make_shared<ShotTimeline>();
  ShotTimeline& timeline = *shared;
  const auto probe = ShotProbe::start(clip, shared, 0);
  for (int i = 0; i < kProbeWait && !probe->finished(); ++i)
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
  const auto shared = std::make_shared<ShotTimeline>();
  ShotTimeline& timeline = *shared;
  const auto probe =
      ShotProbe::start(clip, shared, 30'000'000, ShotProbeOptions{.lead_us = 120'000'000, .pace = 0.0});
  for (int i = 0; i < kProbeWait && !probe->finished(); ++i)
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
  shots.at(1).meta.skins.tones.at(0) = {.hue = 22.0F, .sat = 0.3F};  // la peau de ses visages
  shots.at(1).meta.skins.tones.at(1) = {.hue = 35.0F, .sat = 0.2F};
  shots.at(1).meta.skins.count = 2;
  const std::vector<FlashSpan> flashes{{.from_us = 400'000, .until_us = 1'900'000}};
  REQUIRE(save_shot_cache(dir, key, shots, true, kShotCacheKeep, flashes, 1));
  // Un autre chemin de décodage (logiciel contre NVDEC réduit) ne mesure pas la même image : refait (A42-10).
  CHECK_FALSE(load_shot_cache(dir, key));
  const auto loaded = load_shot_cache(dir, key, 1);
  REQUIRE(loaded);
  CHECK(loaded->flashes == flashes);
  CHECK(loaded->complete);
  REQUIRE(loaded->shots.size() == 2);
  CHECK(loaded->shots.at(0).first.p995 == 0.9F);
  CHECK(loaded->shots.at(1).fade);
  CHECK(loaded->shots.at(1).last.mean == 0.4F);
  CHECK(loaded->shots.at(1).meta == shots.at(1).meta);
  CHECK(loaded->shots.at(1).end_us == 3'000'000);
  {
    // Un fichier d'une autre version de l'analyse (ou du format d'avant, sans version) se refait.
    const std::filesystem::path file = dir / (key + "-00000001.shots");  // un fichier par variante (C42-05)
    std::string bytes;
    {
      std::ifstream in(file, std::ios::binary);
      bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    REQUIRE(bytes.size() > 16);
    std::string other = bytes;
    other.at(8) =
        static_cast<char>(other.at(8) + 1);  // la version de l'analyse, après les huit octets de magie
    std::ofstream(file, std::ios::binary | std::ios::trunc) << other;
    CHECK_FALSE(load_shot_cache(dir, key, 1));
    std::string old = bytes;
    old.at(7) = '3';
    std::ofstream(file, std::ios::binary | std::ios::trunc) << old;
    CHECK_FALSE(load_shot_cache(dir, key, 1));
  }
  // Une analyse arrêtée en route se garde aussi, marquée incomplète.
  REQUIRE(save_shot_cache(dir, key, shots, false));
  CHECK_FALSE(load_shot_cache(dir, key)->complete);
  CHECK_FALSE(load_shot_cache(dir, "0123456789abcdef"));
  // Au-delà de la limite, les plus anciens partent.
  REQUIRE(save_shot_cache(dir, "aaaaaaaaaaaaaaaa", shots, true, 1));
  CHECK_FALSE(load_shot_cache(dir, key));
  CHECK(load_shot_cache(dir, "aaaaaaaaaaaaaaaa"));
  // Une mesure NaN (fichier abîmé) refuse le cache entier.
  shots.at(0).last.p50 = std::numeric_limits<float>::quiet_NaN();
  REQUIRE(save_shot_cache(dir, "bbbbbbbbbbbbbbbb", shots, true));
  CHECK_FALSE(load_shot_cache(dir, "bbbbbbbbbbbbbbbb"));
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

// 8.8, à lancer sur une machine NVIDIA (ctest -R shot_nvdec ou le binaire avec "[.shot_nvdec]") : la
// pré-analyse décodée par NVDEC en 256 × 144 retrouve les dix-huit plans de la logicielle, à une image près,
// et leur APL à 3 %.
TEST_CASE("plans : NVDEC réduit retrouve les plans du logiciel", "[.shot_nvdec][media][shots]") {
  const std::filesystem::path clip = test::require_clip(test::shots_clip());
  const auto analyse = [&](ShotProbeOptions options) {
    const auto shared = std::make_shared<ShotTimeline>();
    ShotTimeline& timeline = *shared;
    const auto probe = ShotProbe::start(clip, shared, 0, std::move(options));
    for (int i = 0; i < kProbeWait && !probe->finished(); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    REQUIRE(probe->finished());
    REQUIRE_FALSE(probe->failure());
    return timeline.shots();
  };
  const std::vector<Shot> software = analyse({});
  const std::vector<Shot> reduced = analyse(ShotProbeOptions{.cuvid_resize = kProbeCuvidSize});
  REQUIRE(software.size() == 18);
  REQUIRE(reduced.size() == software.size());
  for (std::size_t i = 0; i < software.size(); ++i) {
    INFO("plan " << i);
    CHECK(std::abs(reduced[i].start_us - software[i].start_us) <= 42'000);
    CHECK(reduced[i].meta.apl_mean_nits == Approx(software[i].meta.apl_mean_nits).epsilon(0.03));
  }
}
