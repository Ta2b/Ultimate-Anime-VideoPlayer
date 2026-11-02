// L'analyse d'une image pour le grade (kernels/spec/frame_stats.h) : échelles gamma et linéaire,
// centiles, bandes noires, peau, hautes lumières isolées, sources PQ et HLG.
#include "kernels/ref/frame_stats.h"

#include <cmath>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "kernels/ref/transfer.h"

using namespace ajp::kernels;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
namespace fs = ajp::kernels::frame_stats;

namespace {

FrameStats measure(const RgbImage& image, FrameTransfer transfer = FrameTransfer::Sdr) {
  const auto counts = ref::frame_counts(image, transfer);
  REQUIRE(counts);
  return ref::frame_stats_of(*counts);
}

void paint(RgbImage& image, int x0, int y0, int x1, int y1, Rgb color) {
  for (int y = y0; y < y1; ++y) {
    for (int x = x0; x < x1; ++x) {
      image.planes[0].at(x, y) = color.r;
      image.planes[1].at(x, y) = color.g;
      image.planes[2].at(x, y) = color.b;
    }
  }
}

}  // namespace

TEST_CASE("Analyse : gris uniforme SDR, en gamma et en lumière linéaire", "[kernels][frame_stats]") {
  const RgbImage gray(256, 144, 0.5F);
  const FrameStats s = measure(gray);
  const float nits = std::pow(0.5F, 2.4F) * fs::kSdrWhiteNits;
  CHECK_THAT(s.mean, WithinAbs(0.5, 1e-5));
  CHECK(s.luma_hist[32] == 1.0F);
  CHECK_THAT(s.apl_nits, WithinRel(nits, 1e-4F));
  CHECK_THAT(s.maxrgb_mean_nits, WithinRel(nits, 1e-4F));
  // Tous les centiles tombent dans la case du gris : à une case de code PQ près.
  const float code = ref::pq_inverse_eotf(nits);
  for (const float p : s.maxrgb_nits)
    CHECK(std::abs(ref::pq_inverse_eotf(p) - code) <= 1.0F / fs::kLinearBins);
  CHECK(s.bright == 0.0F);
  CHECK(s.highlights == 0.0F);
  CHECK(s.skin == 0.0F);
  CHECK(s.chroma == 0.0F);
  CHECK(s.left == 0.0F);
  CHECK(s.right == 1.0F);
}

TEST_CASE("Analyse : bandes noires écartées, peau et haute lumière isolée", "[kernels][frame_stats]") {
  RgbImage image(256, 144, 0.0F);
  paint(image, 0, 18, 256, 126, {.r = 0.3F, .g = 0.3F, .b = 0.3F});  // bandes de 18 lignes en haut et en bas
  paint(image, 0, 18, 128, 72, {.r = 0.9F, .g = 0.7F, .b = 0.55F});  // un teint clair
  paint(image, 200, 100, 202, 102, {.r = 1.0F, .g = 1.0F, .b = 1.0F});  // un point blanc : une grille
  const FrameStats s = measure(image);
  CHECK_THAT(s.top, WithinAbs(9.0 / 72.0, 1e-6));
  CHECK_THAT(s.bottom, WithinAbs(63.0 / 72.0, 1e-6));
  // Le teint couvre le quart haut gauche du cadre hors bandes (64 × 27 points sur 128 × 54).
  CHECK_THAT(s.skin, WithinAbs(64.0 * 27.0 / (128.0 * 54.0), 1e-6));
  CHECK(s.highlights > 0.0F);
  CHECK(s.highlights <= s.bright);
  CHECK(s.maxrgb_nits.back() >= s.maxrgb_nits[2]);
}

TEST_CASE("Analyse : sources PQ et HLG mesurées en cd/m²", "[kernels][frame_stats]") {
  const RgbImage pq(64, 36, ref::pq_inverse_eotf(1000.0F));
  const FrameStats p = measure(pq, FrameTransfer::Pq);
  CHECK_THAT(p.apl_nits, WithinRel(1000.0F, 1e-3F));
  CHECK(std::abs(ref::pq_inverse_eotf(p.maxrgb_nits[2]) - ref::pq_inverse_eotf(1000.0F)) <=
        1.0F / fs::kLinearBins);
  CHECK(p.bright == 1.0F);
  // HLG blanc (signal 0,75, scène 0,265) : affiché à 1000 × 0,265^1,2 ≈ 203 cd/m².
  const RgbImage hlg(64, 36, 0.75F);
  const FrameStats h = measure(hlg, FrameTransfer::Hlg);
  const float scene = ref::hlg_inverse_oetf(0.75F);
  CHECK_THAT(h.apl_nits, WithinRel(1000.0F * std::pow(scene, 1.2F), 1e-3F));
  // I d'ICtCp : le code PQ de la luminance pour un gris (L = M = Y).
  CHECK_THAT(ref::ictcp_i({.r = 100.0F, .g = 100.0F, .b = 100.0F}),
             WithinAbs(ref::pq_inverse_eotf(100.0F), 1e-4));
}

TEST_CASE("Analyse : image trop petite refusée", "[kernels][frame_stats]") {
  CHECK_FALSE(ref::frame_counts(RgbImage(1, 8), FrameTransfer::Sdr));
}
