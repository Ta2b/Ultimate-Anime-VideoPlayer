// Smooth (« Lisser les dessins tenus », chantier 7.9) : un dessin tenu de deux à six sources, suivi d'un
// dessin nouveau, se fond en un seul mouvement. T3 : u linéaire sur toute la tenue ; l'étage gpu interpole a
// → b à travers la tenue, garde le rythme des sources au-delà de six, et laisse une coupe telle quelle.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "app/play/live_settings.h"
#include "app/play/rife_plan.h"
#include "core/messages.h"
#include "upscale_rig.h"

namespace ajp::app::play {
using namespace upscale_test;

namespace {

// La tenue 1, 2 … n (la source 1 tenue n images), puis n + 1 nouvelle : les paires tenues, et les mesures du
// Smooth contre la première copie (2) jusqu'à la dernière.
void hold(Rig& rig, std::uint32_t n) {
  for (std::uint32_t k = 1; k < n; ++k) rig.backend.score(SourceId{k}, SourceId{k + 1}, 0.0, 10);
  for (std::uint32_t k = 3; k <= n; ++k) rig.backend.score(SourceId{2}, SourceId{k}, 0.0, 10);
}

// RIFE mouvement, ×2, avec ou sans warp après SPAN.
UpscaleSetup smooth_setup(Rig& rig, bool smooth, bool spanned = false) {
  UpscaleSetup s = rig.setup(true);
  s.rife = nullptr;
  s.rife_motion = &rig.motion;
  s.smooth_drawings = smooth;
  s.warp_after_span = spanned;
  return s;
}

// Pousse 1 … last, puis la fin du flux : l'étage, qui lit jusqu'à cinq images en avance, rend tout.
void play(Rig& rig, std::uint32_t last) {
  rig.feed(1, last);
  REQUIRE(rig.stage->post(StreamEnd{.epoch = 0}));
}

bool has_t(const std::vector<float>& ts, double u) {
  return std::ranges::any_of(ts, [u](float t) { return std::abs(static_cast<double>(t) - u) < 1e-6; });
}

}  // namespace

TEST_CASE("Smooth : T3, u linéaire à 1e-9 et continu sur toute la tenue, de deux à six sources",
          "[free][drawings]") {
  for (std::uint32_t n = 2; n <= kSmoothMaxHold; ++n) {
    CAPTURE(n);
    // La tenue va de a (position 0) à b (position n) ; u = position / n, continu d'une paire à la suivante.
    for (int k = 1; k <= 1000 * static_cast<int>(n); ++k) {
      const double position = static_cast<double>(k) / 1000.0;  // ]0, n]
      const auto rank = static_cast<std::uint32_t>(std::ceil(position) - 1.0);
      const double t = position - static_cast<double>(rank);
      CHECK(std::abs(smooth_hold_u(t, rank, n) - position / static_cast<double>(n)) < 1e-9);
    }
    for (std::uint32_t rank = 0; rank + 1 < n; ++rank)
      CHECK(smooth_hold_u(1.0, rank, n) == smooth_hold_u(0.0, rank + 1, n));
    CHECK(smooth_hold_u(1.0, n - 1, n) == 1.0);
  }
  STATIC_REQUIRE(smooth_extra_frames(true) == kSmoothMaxHold - 2);
  STATIC_REQUIRE(smooth_extra_frames(false) == 0);
}

TEST_CASE("Étage gpu : Smooth, la tenue (1, 2) se fond dans le mouvement de 1 vers 3",
          "[app][stage][smooth]") {
  const bool smooth = GENERATE(false, true);
  const bool spanned = GENERATE(false, true);  // warp après SPAN : SPAN de 3 fait en avance, une fois
  INFO("smooth " << smooth << ", warp après SPAN " << spanned);
  Rig rig;
  rig.backend.score(SourceId{1}, SourceId{2}, 0.0, 10);  // 2 tient le dessin de 1 ; 3 est nouveau
  rig.start(smooth_setup(rig, smooth, spanned));
  play(rig, 4);
  const std::size_t expected = smooth ? 7 : 6;
  REQUIRE(rig.collect(expected));
  rig.stage.reset();
  const std::vector<RingFrame> frames = rig.sink.frames();
  // Un SPAN par source montrée ou lue, la tenue 2 reprend celui de 1 : 1, 3, 4.
  if (spanned) CHECK(rig.backend.count("engine span") == 3);
  if (!smooth) {
    CHECK(shape_of(frames) == std::vector<std::string>{"1", "2", "2-3", "3", "3-4", "4"});
    return;
  }
  // La source tenue 2 ne sort pas telle quelle : à son pts, l'image à mi-chemin de 1 vers 3.
  CHECK(shape_of(frames) == std::vector<std::string>{"1", "1-3", "1-3", "2-3", "3", "3-4", "4"});
  CHECK(pts_of(frames) ==
        std::vector<std::int64_t>{40'000, 60'000, 80'000, 100'000, 120'000, 140'000, 160'000});
  // Le warp de la paire (1, 3) : 1/4 et 1/2 sur la passe du milieu. La paire (2, 3) reprend à 3/4 : RIFE y
  // estime l'instant lui-même (une seule image, pas de passe au milieu qui la serve).
  const std::vector<float> warps = rig.backend.warp_ts();
  REQUIRE(warps.size() >= 2);
  CHECK(warps[0] == 0.25F);
  CHECK(warps[1] == 0.5F);
  const std::vector<float> rife = rig.backend.rife_ts();
  CHECK(std::ranges::find(rife, 0.75F) != rife.end());
}

TEST_CASE("Étage gpu : Smooth, une tenue de trois à six sources se fond dans un seul mouvement",
          "[app][stage][smooth]") {
  const std::uint32_t n = GENERATE(3U, 4U, 5U, 6U);
  const bool spanned = GENERATE(false, true);
  INFO("tenue de " << n << " sources, warp après SPAN " << spanned);
  Rig rig;
  hold(rig, n);
  rig.start(smooth_setup(rig, true, spanned));
  const std::uint32_t b = n + 1;
  play(rig, b + 1);
  // 1 seule, puis deux images par paire de tête, la dernière paire (n, b), b, et la paire (b, b + 1).
  const std::size_t expected = 1 + 2 * (n - 1) + 2 + 2;
  REQUIRE(rig.collect(expected));
  rig.stage.reset();
  const std::vector<RingFrame> frames = rig.sink.frames();
  std::vector<std::string> shape{"1"};
  for (std::uint32_t k = 1; k < n; ++k) {
    const std::string middle = std::to_string(k) + "-" + std::to_string(b);
    shape.insert(shape.end(), {middle, middle});  // aucune copie ne sort telle quelle
  }
  shape.insert(shape.end(), {std::to_string(n) + "-" + std::to_string(b), std::to_string(b),
                             std::to_string(b) + "-" + std::to_string(b + 1), std::to_string(b + 1)});
  CHECK(shape_of(frames) == shape);
  // Chaque sortie à son instant, à la demi-période : le rythme de la sortie ne bouge pas.
  const std::vector<std::int64_t> pts = pts_of(frames);
  for (std::size_t i = 0; i < pts.size(); ++i)
    CHECK(pts[i] == kPeriodUs + static_cast<std::int64_t>(i) * 20'000);
  // Les instants de la tenue : (k + 1/2) / n et (k + 1) / n, jusqu'à la dernière demi-paire (n - 1/2) / n.
  std::vector<float> ts = rig.backend.warp_ts();
  const std::vector<float> rife = rig.backend.rife_ts();
  ts.insert(ts.end(), rife.begin(), rife.end());
  for (std::uint32_t k = 0; k < n; ++k) {
    CAPTURE(k);
    CHECK(has_t(ts, (k + 0.5) / n));
    if (k + 1 < n) CHECK(has_t(ts, (k + 1.0) / n));
  }
  // Toutes les paires de tête lisent b, jamais la copie suivante.
  for (const auto& [left, right] : rig.backend.rife_pairs()) {
    if (left >= 1 && left < n) CHECK(right == b);
  }
  // Warp après SPAN : SPAN de 1, de b une fois en avance, et de b + 1 ; les copies reprennent celui de 1.
  if (spanned) CHECK(rig.backend.count("engine span") == 3);
  // Un dessin nouveau cherché derrière la suivante : une mesure par copie au-delà de la deuxième, et b.
  CHECK(rig.backend.count("scores_now 2 " + std::to_string(b)) == 1);
}

TEST_CASE("Étage gpu : Smooth, une tenue de sept sources garde la copie", "[app][stage][smooth]") {
  Rig rig;
  hold(rig, 7);
  rig.start(smooth_setup(rig, true));
  play(rig, 9);
  REQUIRE(rig.collect(11));
  rig.stage.reset();
  CHECK(shape_of(rig.sink.frames()) ==
        std::vector<std::string>{"1", "2", "3", "4", "5", "6", "7", "7-8", "8", "8-9", "9"});
}

TEST_CASE("Étage gpu : Smooth, une coupe au bout de la tenue garde le rythme des sources",
          "[app][stage][smooth]") {
  Rig rig;
  hold(rig, 3);
  rig.backend.score(SourceId{2}, SourceId{4}, 0.9, 1000);  // 4 ouvre un autre plan
  rig.backend.score(SourceId{3}, SourceId{4}, 0.9, 1000);
  rig.start(smooth_setup(rig, true));
  play(rig, 5);
  REQUIRE(rig.collect(6));
  rig.stage.reset();
  CHECK(shape_of(rig.sink.frames()) == std::vector<std::string>{"1", "2", "3", "4", "4-5", "5"});
}

TEST_CASE("Étage gpu : Smooth éteint, une tenue de trois sources garde son rythme", "[app][stage][smooth]") {
  Rig rig;
  hold(rig, 3);
  rig.start(smooth_setup(rig, false));
  play(rig, 5);
  REQUIRE(rig.collect(7));
  rig.stage.reset();
  CHECK(shape_of(rig.sink.frames()) == std::vector<std::string>{"1", "2", "3", "3-4", "4", "4-5", "5"});
}

TEST_CASE("Étage gpu : Smooth, la fin du flux au milieu d'une tenue garde la copie", "[app][stage][smooth]") {
  Rig rig;
  hold(rig, 4);
  rig.start(smooth_setup(rig, true));
  play(rig, 4);  // b n'arrive jamais
  REQUIRE(rig.collect(4));
  rig.stage.reset();
  CHECK(shape_of(rig.sink.frames()) == std::vector<std::string>{"1", "2", "3", "4"});
}

}  // namespace ajp::app::play
