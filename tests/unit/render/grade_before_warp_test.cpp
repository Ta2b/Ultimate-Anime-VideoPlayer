// Le grade avant le warp (app/play/source_grade.h) face au grade de l'image affichée, sur le GPU : une image
// intermédiaire est un mélange de pixels voisins de ses deux sources (lectures bilinéaires du warp, masque de
// RIFE) ; le grade d'un mélange n'est pas le mélange des grades. L'écart se mesure ici en codes PQ 10 bits
// sur des plans en mouvement : un panoramique, un zoom, et le fantôme d'un mouvement que le flot n'a pas
// suivi (flot nul, deux images différentes mêlées à moitié), le pire cas.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "grade_chain_support.h"
#include "render/grade_params.h"

namespace {

using namespace ajp::render;
using namespace ajp::render::gpu_test;

constexpr std::size_t kPixels = static_cast<std::size_t>(kSide) * kSide;

// Une image source, RGB en gamma 2,4 de 0 à 1 (ce que SPAN rend), kSide × kSide.
using Image = std::vector<float>;

float at(const Image& img, int x, int y, int c) {
  x = std::clamp(x, 0, kSide - 1);
  y = std::clamp(y, 0, kSide - 1);
  return img[(static_cast<std::size_t>(y) * kSide + static_cast<std::size_t>(x)) * 3 +
             static_cast<std::size_t>(c)];
}

// Lecture bilinéaire, bords répétés, comme le warp.
float sample(const Image& img, double x, double y, int c) {
  const double fx = std::floor(x);
  const double fy = std::floor(y);
  const auto ix = static_cast<int>(fx);
  const auto iy = static_cast<int>(fy);
  const auto ax = static_cast<float>(x - fx);
  const auto ay = static_cast<float>(y - fy);
  const float top = at(img, ix, iy, c) * (1.0F - ax) + at(img, ix + 1, iy, c) * ax;
  const float bottom = at(img, ix, iy + 1, c) * (1.0F - ax) + at(img, ix + 1, iy + 1, c) * ax;
  return top * (1.0F - ay) + bottom * ay;
}

// Où le warp lit chaque source pour l'image intermédiaire d'un pixel, et le poids de a (le masque).
struct Motion {
  std::function<std::array<double, 2>(double, double)> from_a;
  std::function<std::array<double, 2>(double, double)> from_b;
  float mask = 0.5F;
};

Image warp(const Image& a, const Image& b, const Motion& m) {
  Image out(kPixels * 3);
  for (int y = 0; y < kSide; ++y) {
    for (int x = 0; x < kSide; ++x) {
      const auto pa = m.from_a(x, y);
      const auto pb = m.from_b(x, y);
      for (int c = 0; c < 3; ++c) {
        out[(static_cast<std::size_t>(y) * kSide + static_cast<std::size_t>(x)) * 3 +
            static_cast<std::size_t>(c)] =
            m.mask * sample(a, pa[0], pa[1], c) + (1.0F - m.mask) * sample(b, pb[0], pb[1], c);
      }
    }
  }
  return out;
}

// Une source mise en PQ BT.2020 comme l'écran la donne au grade (relevée à 203 nits × kAnimeLift).
std::vector<std::uint16_t> to_pq(const Image& img) {
  std::vector<std::uint16_t> px(kPixels * 4);
  for (std::size_t i = 0; i < kPixels; ++i) {
    std::array<double, 3> lin{};
    for (std::size_t c = 0; c < 3; ++c)
      lin[c] = std::pow(std::clamp(static_cast<double>(img[3 * i + c]), 0.0, 1.0), 2.4) * 203.0 * kAnimeLift;
    const std::array<double, 3> pq{
        nits_to_pq((0.6274040 * lin[0]) + (0.3292820 * lin[1]) + (0.0433136 * lin[2])),
        nits_to_pq((0.0690970 * lin[0]) + (0.9195400 * lin[1]) + (0.0113612 * lin[2])),
        nits_to_pq((0.0163916 * lin[0]) + (0.0880132 * lin[1]) + (0.8955950 * lin[2]))};
    for (std::size_t c = 0; c < 3; ++c)
      px[4 * i + c] = static_cast<std::uint16_t>(std::lround(pq[c] * 65535.0));
    px[4 * i + 3] = 65535;
  }
  return px;
}

// Une image gradée (PQ 16 bits) relue dans le tenseur fp16 de SPAN : 11 bits de mantisse.
Image graded_tensor(const std::vector<std::uint16_t>& px) {
  Image out(kPixels * 3);
  for (std::size_t i = 0; i < kPixels; ++i) {
    for (std::size_t c = 0; c < 3; ++c) {
      const double v = px[4 * i + c] / 65535.0;
      int e = 0;
      const double m = std::frexp(v, &e);
      out[3 * i + c] = static_cast<float>(std::ldexp(std::round(m * 2048.0) / 2048.0, e));
    }
  }
  return out;
}

struct Gaps {
  double max = 0.0;
  double p999 = 0.0;
  double p99 = 0.0;
  double mean = 0.0;
};

void add_gaps(std::vector<double>& gaps, const std::vector<std::uint16_t>& per_display, const Image& before) {
  for (std::size_t i = 0; i < kPixels; ++i) {
    double gap = 0.0;
    for (std::size_t c = 0; c < 3; ++c) {
      const double shown = std::round(std::clamp(static_cast<double>(before[3 * i + c]), 0.0, 1.0) * 65535.0);
      gap = std::max(gap, 1023.0 / 65535.0 * std::abs(double(per_display[4 * i + c]) - shown));
    }
    gaps.push_back(gap);
  }
}

Gaps summary(std::vector<double> gaps) {
  std::ranges::sort(gaps);
  Gaps out;
  out.max = gaps.back();
  out.p999 = gaps[gaps.size() * 999 / 1000];
  out.p99 = gaps[gaps.size() * 99 / 100];
  for (const double gap : gaps) out.mean += gap / static_cast<double>(gaps.size());
  return out;
}

// L'écart, sur une paire, de warp(G(a), G(b)) à G(warp(a, b)) ; et sans grade, celui du seul passage du
// mélange du gamma de la source au PQ de la sortie (PQ(warp(a, b)) face à warp(PQ(a), PQ(b))), la part de
// l'écart que tout grade en PQ donnerait.
// Le grade d'une image tenue : sa zone lissée dans le temps (2-area.hook, ~0,1 s) repartie de l'image d'avant
// a rejoint sa mesure après quelques rendus (0,49 de reste par image à 24 i/s). Chaque image est ainsi
// gradée comme dans un plan, sans la trace de celle mesurée juste avant.
std::vector<std::uint16_t> steady(Gpu& gpu, const std::vector<std::uint16_t>& px, const GradeParams& params) {
  for (int i = 0; i < 7; ++i) (void)graded_image(gpu, px, params);
  return graded_image(gpu, px, params);
}

struct PairGaps {
  std::vector<double> graded;
  std::vector<double> plain;
};

void pair_gaps(Gpu& gpu, const GradeParams& params, const Image& a, const Image& b, const Motion& m,
               PairGaps& gaps) {
  const std::vector<std::uint16_t> mixed = to_pq(warp(a, b, m));
  const std::vector<std::uint16_t> pa = to_pq(a);
  const std::vector<std::uint16_t> pb = to_pq(b);
  add_gaps(gaps.graded, steady(gpu, mixed, params),
           warp(graded_tensor(steady(gpu, pa, params)), graded_tensor(steady(gpu, pb, params)), m));
  add_gaps(gaps.plain, mixed, warp(graded_tensor(pa), graded_tensor(pb), m));
}

// Les trois mouvements à mi-chemin (t = 0,5) entre a et b, b tiré de a.
struct Moves {
  Image pan;   // b : a décalée de (5,5 ; 2,25) pixels
  Image zoom;  // b : a agrandie de 4 % autour du centre
};

constexpr double kPanX = 5.5;
constexpr double kPanY = 2.25;
constexpr double kZoom = 1.04;

Moves moves_of(const Image& a) {
  Moves out{.pan = Image(kPixels * 3), .zoom = Image(kPixels * 3)};
  constexpr double kMid = (kSide - 1) / 2.0;
  for (int y = 0; y < kSide; ++y) {
    for (int x = 0; x < kSide; ++x) {
      for (int c = 0; c < 3; ++c) {
        const std::size_t i = (static_cast<std::size_t>(y) * kSide + static_cast<std::size_t>(x)) * 3 +
                              static_cast<std::size_t>(c);
        out.pan[i] = sample(a, x - kPanX, y - kPanY, c);
        out.zoom[i] = sample(a, kMid + (x - kMid) / kZoom, kMid + (y - kMid) / kZoom, c);
      }
    }
  }
  return out;
}

Motion pan_motion() {
  return {.from_a = [](double x, double y) { return std::array{x - kPanX / 2, y - kPanY / 2}; },
          .from_b = [](double x, double y) { return std::array{x + kPanX / 2, y + kPanY / 2}; }};
}

Motion zoom_motion() {
  constexpr double kMid = (kSide - 1) / 2.0;
  const double half = std::sqrt(kZoom);
  return {
      .from_a = [half](double x,
                       double y) { return std::array{kMid + (x - kMid) / half, kMid + (y - kMid) / half}; },
      .from_b = [half](double x,
                       double y) { return std::array{kMid + (x - kMid) * half, kMid + (y - kMid) * half}; }};
}

Motion ghost_motion() {
  return {.from_a = [](double x, double y) { return std::array{x, y}; },
          .from_b = [](double x, double y) { return std::array{x, y}; }};
}

// Une image façon cellulo : un ciel en dégradé, des aplats ronds aux couleurs saturées, cernés de noir.
Image cel_image(std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> unit(0.0F, 1.0F);
  Image img(kPixels * 3);
  for (int y = 0; y < kSide; ++y) {
    for (int x = 0; x < kSide; ++x) {
      const float t = static_cast<float>(y) / kSide;
      const std::size_t i = (static_cast<std::size_t>(y) * kSide + static_cast<std::size_t>(x)) * 3;
      img[i] = 0.35F + 0.4F * t;
      img[i + 1] = 0.6F + 0.3F * t;
      img[i + 2] = 0.95F;
    }
  }
  for (int blob = 0; blob < 14; ++blob) {
    const float cx = unit(rng) * kSide;
    const float cy = unit(rng) * kSide;
    const float r = 12.0F + unit(rng) * 40.0F;
    const std::array<float, 3> colour{unit(rng), unit(rng), unit(rng)};
    for (int y = 0; y < kSide; ++y) {
      for (int x = 0; x < kSide; ++x) {
        const float d = std::hypot(static_cast<float>(x) - cx, static_cast<float>(y) - cy);
        if (d > r) continue;
        const std::size_t i = (static_cast<std::size_t>(y) * kSide + static_cast<std::size_t>(x)) * 3;
        for (std::size_t c = 0; c < 3; ++c) img[i + c] = d > r - 2.0F ? 0.05F : colour[c];
      }
    }
  }
  return img;
}

struct Report {
  Gaps pan;
  Gaps zoom;
  Gaps ghost;
  Gaps pan_plain;
  Gaps zoom_plain;
  Gaps ghost_plain;
};

Report measure(Gpu& gpu, const std::vector<Image>& stills) {
  const Pile pile(0, 0, true);
  set_player_output(gpu, pile, kPeak);
  GradeParams params = player_params(kPeak);
  params.look_mixes = pile.mixes;
  PairGaps pan;
  PairGaps zoom;
  PairGaps ghost;
  for (std::size_t s = 0; s < stills.size(); ++s) {
    const Image& a = stills[s];
    const Moves moved = moves_of(a);
    pair_gaps(gpu, params, a, moved.pan, pan_motion(), pan);
    pair_gaps(gpu, params, a, moved.zoom, zoom_motion(), zoom);
    pair_gaps(gpu, params, a, stills[(s + 1) % stills.size()], ghost_motion(), ghost);
  }
  return {.pan = summary(pan.graded),
          .zoom = summary(zoom.graded),
          .ghost = summary(ghost.graded),
          .pan_plain = summary(pan.plain),
          .zoom_plain = summary(zoom.plain),
          .ghost_plain = summary(ghost.plain)};
}

void print(const char* what, const Gaps& g) {
  std::printf("%s : écart maximal %.2f codes PQ 10 bits, 99,9 %% sous %.2f, 99 %% sous %.2f, moyen %.3f\n",
              what, g.max, g.p999, g.p99, g.mean);
}

void print(const Report& r) {
  print("panoramique", r.pan);
  print("zoom", r.zoom);
  print("fantôme (flot nul, deux images)", r.ghost);
  print("sans grade, panoramique", r.pan_plain);
  print("sans grade, zoom", r.zoom_plain);
  print("sans grade, fantôme", r.ghost_plain);
}

TEST_CASE("grade avant le warp : l'écart au grade de l'image affichée, plans en mouvement façon cellulo",
          "[render][gpu][chain]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const Report report = measure(gpu, {cel_image(1), cel_image(2), cel_image(3)});
  print(report);
  // Des gardes autour de la mesure sur lavapipe (docs/hdr-anime.md, « Le grade avant le warp ») : ces aplats
  // saturés cernés de noir sont plus durs que les plans réels, où le flot juste donne 0,8 code en moyenne.
  CHECK(report.pan.mean < 4.0);
  CHECK(report.zoom.mean < 4.0);
  CHECK(report.ghost.mean < 40.0);
  // Une image tenue (a = b, flot nul) : le même grade, à l'arrondi du tenseur fp16 près.
  const Pile pile(0, 0, true);
  set_player_output(gpu, pile, kPeak);
  GradeParams params = player_params(kPeak);
  params.look_mixes = pile.mixes;
  PairGaps still;
  pair_gaps(gpu, params, cel_image(4), cel_image(4), ghost_motion(), still);
  CHECK(summary(still.graded).max <= 0.5);
}

// Les images réelles : les plans de data/calibration/calibration.mkv, découpés en kSide × kSide par
// tools/grade/warp_stills.sh (RGB 8 bits brut, à la file), dans le dossier temporaire. Sans eux, sautée.
TEST_CASE("mesure : grade avant le warp sur les plans de calibration", "[.mesure]") {
  const std::filesystem::path file = std::filesystem::temp_directory_path() / "ajp-warp-stills.rgb";
  std::ifstream in(file, std::ios::binary);
  if (!in) SKIP("pas de " + file.string() + " : lancer tools/grade/warp_stills.sh");
  std::vector<Image> stills;
  std::vector<unsigned char> raw(kPixels * 3);
  while (in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()))) {
    Image img(raw.size());
    std::ranges::transform(raw, img.begin(), [](unsigned char v) { return static_cast<float>(v) / 255.0F; });
    stills.push_back(std::move(img));
  }
  REQUIRE(stills.size() >= 2);
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  std::printf("%zu découpes\n", stills.size());
  print(measure(gpu, stills));
}

}  // namespace
