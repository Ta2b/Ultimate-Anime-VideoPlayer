// Le rythme des plans (media/shot_rhythm.h) : la paire mesurée sur la grille, le flux, les sommes d'un plan
// dans le découpeur, et leur passage par le cache des plans.
#include "media/shot_rhythm.h"

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "media/shot_cache.h"
#include "media/shot_cutter.h"

using namespace kirari::media;
using Catch::Approx;

namespace {

constexpr int kW = 128;
constexpr int kH = 72;
constexpr std::int64_t kFrame = 41'667;  // 24 im/s

std::vector<float> plane(float value) {
  return std::vector<float>(static_cast<std::size_t>(kW) * kH, value);
}

// Une image du même plan pour le découpeur : même lumière, seul son rythme change.
ShotFrame frame(std::int64_t index, std::optional<RhythmPair> rhythm) {
  ShotFrame f;
  f.pts_us = index * kFrame;
  f.histogram.at(32) = 1.0F;
  f.grid.fill(0.5F);
  f.mean = 0.5F;
  f.rhythm = rhythm;
  return f;
}

}  // namespace

TEST_CASE("rythme : paire, flux, plan en deux et cache", "[media][shots]") {
  // Une paire : la moitié gauche bouge de 20 codes, la droite garde un grain de 1 code.
  std::vector<float> a = plane(0.5F);
  std::vector<float> b = a;
  for (int y = 0; y < kH; ++y) {
    for (int x = 0; x < kW; ++x)
      b[static_cast<std::size_t>(y * kW + x)] += (x < kW / 2 ? 20.0F : 1.0F) / 255.0F;
  }
  const std::optional<RhythmPair> pair = rhythm_pair_of(a, b, kW, kH);
  REQUIRE(pair);
  CHECK(pair->changed == Approx(0.5F));
  CHECK(pair->grain == Approx(1.0F).margin(0.01));
  CHECK_FALSE(held_pair(*pair));
  CHECK(strong_pair(*pair));  // la moitié de l'image : un grand changement, sans flux
  CHECK_FALSE(rhythm_pair_of(a, std::vector<float>(3, 0.0F), kW, kH));

  // Le flux sépare le panoramique (grand changement, dessins immobiles face à la caméra) du sakuga.
  const RhythmPair pan = with_flow(*pair, {.shift_x = 6.0F, .residual = 0.1F}, 144);
  CHECK(pan.flow);
  CHECK(pan.camera == Approx(6.0F / 144.0F));
  CHECK_FALSE(held_pair(pan));
  CHECK_FALSE(strong_pair(pan));
  CHECK(strong_pair(with_flow(*pair, {.residual = 9.0F}, 144)));
  CHECK(held_pair(with_flow(*pair, {.shift_x = 0.01F, .residual = 0.01F}, 144)));

  // La sonde : rien pour la première image ni après un saut.
  RhythmProbe probe;
  CHECK_FALSE(probe.push(a, kW, kH));
  CHECK(probe.push(b, kW, kH));
  probe.reset();
  CHECK_FALSE(probe.push(a, kW, kH));

  // Un plan en deux : un dessin neuf une image sur deux. La première paire du plan (celle de sa première
  // image) n'entre pas : elle enjambe la coupe.
  const RhythmPair drawn{.changed = 0.2F, .grain = 0.5F};
  const RhythmPair held{.changed = 0.0F, .grain = 0.5F};
  ShotCutter cutter;
  std::vector<Shot> shots;
  const RhythmPair crossing{.changed = 1.0F, .grain = 9.0F};
  for (std::int64_t i = 0; i < 48; ++i) {
    const std::optional<RhythmPair> r =
        i == 0 ? std::optional(crossing) : std::optional(i % 2 == 0 ? drawn : held);
    if (auto s = cutter.push(frame(i, r))) shots.push_back(*s);
  }
  for (const Shot& s : cutter.finish(48 * kFrame)) shots.push_back(s);
  REQUIRE(shots.size() == 1);
  const ShotRhythm& rhythm = shots.front().rhythm;
  CHECK(rhythm.pairs == 47);
  CHECK(rhythm.held == 24);
  CHECK(drawing_cadence(rhythm) == Approx(47.0F / 23.0F));
  CHECK(rhythm.changed == Approx(0.2F));
  CHECK(rhythm.grain == Approx(0.5F));
  CHECK(rhythm.strong == 0);

  // Le cache garde le rythme.
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "kirari-shot-rhythm";
  std::filesystem::remove_all(dir);
  const std::string key = "dddddddddddddddd";
  REQUIRE(save_shot_cache(dir, key, shots, true));
  const std::optional<CachedShots> back = load_shot_cache(dir, key);
  REQUIRE(back);
  REQUIRE(back->shots.size() == 1);
  CHECK(back->shots.front().rhythm == rhythm);
  std::filesystem::remove_all(dir);
}
