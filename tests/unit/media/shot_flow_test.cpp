// Le flux optique de NVIDIA dans la pré-analyse des plans (media/probe_flow.h) : la part des blocs sans
// correspondance tirée des vecteurs, la luma réduite d'une image décodée, et ce que le découpeur en fait
// (media/shot_cutter.h). Sans GPU : les vecteurs sont donnés, comme NVOF les rendrait.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

extern "C" {
#include <libavutil/frame.h>
}

#include "media/frame_data.h"
#include "media/probe_flow.h"
#include "media/shot_cutter.h"

using namespace kirari::media;
using Catch::Approx;

namespace {

constexpr std::size_t kLuma = static_cast<std::size_t>(kProbeFlowWidth) * kProbeFlowHeight;
constexpr std::size_t kCells = static_cast<std::size_t>(kProbeFlowCellsX) * kProbeFlowCellsY;
constexpr std::int64_t kFrame = 41'667;  // 24 im/s

ProbeLuma noise(unsigned seed) {
  std::mt19937 random(seed);
  std::uniform_int_distribution<int> value(30, 220);
  ProbeLuma out(kLuma);
  for (auto& v : out) v = static_cast<std::uint8_t>(value(random));
  return out;
}

// luma décalée de dx points vers la droite (bords répétés).
ProbeLuma shifted(const ProbeLuma& luma, int dx) {
  ProbeLuma out(kLuma);
  for (int y = 0; y < kProbeFlowHeight; ++y)
    for (int x = 0; x < kProbeFlowWidth; ++x)
      out[static_cast<std::size_t>(y * kProbeFlowWidth + x)] =
          luma[static_cast<std::size_t>(y * kProbeFlowWidth + std::clamp(x - dx, 0, kProbeFlowWidth - 1))];
  return out;
}

std::vector<FlowVector> uniform_flow(int dx_px, int dy_px) {
  return std::vector<FlowVector>(kCells, FlowVector{.x = static_cast<std::int16_t>(dx_px * 32),
                                                    .y = static_cast<std::int16_t>(dy_px * 32)});
}

// Une image de luminance luma et de bleu blue : deux dessins d'un même plan diffèrent par le bleu, de quoi
// être distincts (au-delà d'un dessin tenu) sans approcher le seuil d'une coupe.
ShotFrame drawing(std::int64_t index, float luma, float blue, std::optional<float> unmatched) {
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
  f.blue = blue;
  if (unmatched) f.motion = ProbeMotion{.unmatched = *unmatched, .textured = 0.6F};
  return f;
}

std::vector<Shot> cut(const std::vector<ShotFrame>& frames) {
  ShotCutter cutter;
  std::vector<Shot> shots;
  for (const ShotFrame& f : frames)
    if (auto s = cutter.push(f)) shots.push_back(*s);
  for (const Shot& s : cutter.finish(frames.back().pts_us + kFrame)) shots.push_back(s);
  return shots;
}

// Un plan de count images à partir de first, animé sur deux : un nouveau dessin toutes les deux images
// (bleu base, puis base + 0,01), la part sans correspondance drawn sur chaque nouveau dessin, 0 sur les
// images tenues.
void animate(std::vector<ShotFrame>& frames, std::int64_t first, int count, float luma, float base,
             std::optional<float> drawn) {
  for (int i = 0; i < count; ++i) {
    const bool fresh = i % 2 == 0;
    const float blue = base + ((i / 2) % 2 == 0 ? 0.0F : 0.01F);
    std::optional<float> unmatched = drawn;
    if (drawn && !fresh) unmatched = 0.0F;
    frames.push_back(drawing(first + i, luma, blue, unmatched));
  }
}

}  // namespace

TEST_CASE("motion_of : la même image se retrouve partout, immobile") {
  const ProbeLuma a = noise(1);
  const auto motion = motion_of(a, a, uniform_flow(0, 0));
  REQUIRE(motion);
  CHECK(motion->unmatched == 0.0F);
  CHECK(motion->textured == Approx(1.0F));
  CHECK(motion->still == Approx(1.0F));
  CHECK(motion->mean_motion == 0.0F);
}

TEST_CASE("motion_of : un panoramique suivi par ses vecteurs, dans l'un ou l'autre sens") {
  const ProbeLuma a = noise(2);
  const ProbeLuma b = shifted(a, 3);
  for (const int sign : {-1, 1}) {
    const auto motion = motion_of(a, b, uniform_flow(3 * sign, 0));
    REQUIRE(motion);
    CHECK(motion->unmatched < 0.03F);  // les blocs du bord gauche, dont le contenu entre dans l'image
    CHECK(std::abs(motion->shift_x) == Approx(3.0F));
    CHECK(motion->shift_y == 0.0F);
    CHECK(motion->mean_motion == Approx(3.0F).margin(0.1));
    CHECK(motion->residual == Approx(0.0F).margin(0.1));
    CHECK(motion->still < 0.05F);
  }
  // Sans les vecteurs, le même panoramique ne se retrouve pas.
  const auto blind = motion_of(a, b, uniform_flow(0, 0));
  REQUIRE(blind);
  CHECK(blind->unmatched > 0.8F);
}

TEST_CASE("motion_of : une coupe ne se retrouve pas, un éclair sur le même dessin si") {
  const ProbeLuma a = noise(3);
  const auto cut_motion = motion_of(a, noise(4), uniform_flow(0, 0));
  REQUIRE(cut_motion);
  CHECK(cut_motion->unmatched > 0.9F);
  // Le même dessin éclairci de 30 niveaux : le contenu centré se retrouve.
  ProbeLuma lit = a;
  for (auto& v : lit) v = static_cast<std::uint8_t>(v + 30);
  const auto flash = motion_of(a, lit, uniform_flow(0, 0));
  REQUIRE(flash);
  CHECK(flash->unmatched == 0.0F);
}

TEST_CASE("motion_of : rien à juger sur une image unie ou de mauvaise taille") {
  const ProbeLuma flat(kLuma, 128);
  CHECK_FALSE(motion_of(flat, flat, uniform_flow(0, 0)));
  const ProbeLuma a = noise(5);
  CHECK_FALSE(motion_of(a, ProbeLuma(kLuma - 1, 0), uniform_flow(0, 0)));
  CHECK_FALSE(motion_of(a, a, std::vector<FlowVector>(kCells - 1)));
}

TEST_CASE("probe_luma_of : la luma ramenée en 256 × 144 par moyenne de zones") {
  FramePtr f(av_frame_alloc());
  f->width = 2 * kProbeFlowWidth;
  f->height = 2 * kProbeFlowHeight;
  f->format = AV_PIX_FMT_NV12;
  REQUIRE(av_frame_get_buffer(f.get(), 0) == 0);
  for (int y = 0; y < f->height; ++y)
    for (int x = 0; x < f->width; ++x)
      f->data[0][(y * f->linesize[0]) + x] = static_cast<std::uint8_t>(((x / 2) + (x % 2) * 2) % 200);
  for (int y = 0; y < f->height / 2; ++y)
    for (int x = 0; x < f->width; ++x) f->data[1][(y * f->linesize[1]) + x] = 128;
  auto decoded = to_decoded_frame(std::move(f), 0, kFrame);
  REQUIRE(decoded);
  const auto luma = probe_luma_of(*decoded);
  REQUIRE(luma);
  REQUIRE(luma->size() == kLuma);
  // Chaque point moyenne deux colonnes, x mod 200 et x mod 200 + 2 : x mod 200 + 1.
  CHECK((*luma)[10] == 11);
  CHECK((*luma)[static_cast<std::size_t>(100 * kProbeFlowWidth) + 150] == 151);
}

TEST_CASE("Découpe : sans le flux, rien ne change") {
  std::vector<ShotFrame> with;
  std::vector<ShotFrame> without;
  animate(with, 0, 24, 0.3F, 0.0F, 0.1F);
  animate(with, 24, 24, 0.7F, 0.0F, 0.9F);
  animate(without, 0, 24, 0.3F, 0.0F, std::nullopt);
  animate(without, 24, 24, 0.7F, 0.0F, std::nullopt);
  CHECK(cut(without).size() == 2);
  CHECK(cut(with).size() == 2);  // une vraie coupe, que le flux confirme
}

TEST_CASE("Découpe : un saut de lumière que le flux retrouve n'est pas une coupe") {
  std::vector<ShotFrame> frames;
  animate(frames, 0, 24, 0.3F, 0.0F, 0.1F);
  animate(frames, 24, 24, 0.7F, 0.0F, 0.05F);  // le même dessin, plus clair : ses blocs se retrouvent
  CHECK(cut(frames).size() == 1);
  // Le même saut quand le flux ne voit pas assez de détail : jugé comme avant.
  for (ShotFrame& f : frames)
    if (f.motion) f.motion->textured = 0.1F;
  CHECK(cut(frames).size() == 2);
}

TEST_CASE("Découpe : deux plans aux mêmes couleurs, séparés par le flux") {
  std::vector<ShotFrame> frames;
  animate(frames, 0, 24, 0.3F, 0.0F, 0.1F);
  animate(frames, 24, 24, 0.3F, 0.04F, 0.1F);
  frames.at(24).motion->unmatched = 0.9F;  // le premier dessin du second plan
  std::vector<ShotFrame> blind = frames;
  for (ShotFrame& f : blind) f.motion.reset();
  REQUIRE(cut(blind).size() == 1);  // la lumière seule ne la voit pas
  const auto shots = cut(frames);
  REQUIRE(shots.size() == 2);
  CHECK(shots.at(1).start_us == 24 * kFrame);
  CHECK(shots.at(1).cut_before);
}

TEST_CASE("Découpe : un plan fixe après une coupe vue par le flux la confirme") {
  std::vector<ShotFrame> frames;
  animate(frames, 0, 24, 0.3F, 0.0F, 0.1F);
  frames.push_back(drawing(24, 0.3F, 0.04F, 0.9F));
  for (int i = 25; i < 48; ++i) frames.push_back(drawing(i, 0.3F, 0.04F, 0.0F));  // le même dessin, tenu
  const auto shots = cut(frames);
  REQUIRE(shots.size() == 2);
  CHECK(shots.at(1).start_us == 24 * kFrame);
}

TEST_CASE("Découpe : un combat que le flux ne suit plus n'ouvre pas de plan") {
  std::vector<ShotFrame> frames;
  animate(frames, 0, 24, 0.3F, 0.0F, 0.1F);
  animate(frames, 24, 24, 0.3F, 0.0F, 0.9F);  // chaque nouveau dessin, sur deux, perdu par le flux
  CHECK(cut(frames).size() == 1);
}

TEST_CASE("Découpe : un fondu enchaîné, perdu par le flux sur plusieurs dessins, n'ouvre pas de plan") {
  std::vector<ShotFrame> frames;
  animate(frames, 0, 24, 0.3F, 0.0F, 0.1F);
  for (int i = 0; i < 12; ++i)
    frames.push_back(drawing(24 + i, 0.3F + (0.01F * static_cast<float>(i)), 0.0F, 0.7F));
  animate(frames, 36, 24, 0.42F, 0.0F, 0.1F);
  const auto shots = cut(frames);
  CHECK(shots.size() == 1);
}
