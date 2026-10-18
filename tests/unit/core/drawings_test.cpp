#include "core/drawings.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <string_view>

#include <catch2/catch_test_macros.hpp>

#include "core/cadence.h"

using ajp::RefusalCode;
using ajp::SourceId;
using namespace ajp::core;

namespace {

constexpr std::uint32_t kFirst = 100;  // identités loin de SourceId{} par défaut

SourceId id(std::uint32_t position) {
  return {kFirst + position};
}

// Une mesure par lettre : N nouveau dessin, h dessin tenu (bruit de 0,40),
// i doublon exact, C coupe de scène.
PairMeasure measure(char c) {
  switch (c) {
    case 'h':
      return {.worst_block_diff = 0.40};
    case 'i':
      return {.identical = true};
    case 'C':
      return {.worst_block_diff = 50.0, .scene_cut = true};
    default:
      return {.worst_block_diff = 3.0};
  }
}

// La première lettre est la source de start() ; les suivantes sont mesurées.
DrawingsTimeline timeline(std::string_view pattern, bool finished = true) {
  DrawingsTimeline drawings;
  drawings.start(id(0));
  for (std::uint32_t k = 1; k < pattern.size(); ++k) REQUIRE(drawings.push(id(k), measure(pattern[k])));
  if (finished) drawings.finish();
  return drawings;
}

InstantDecision decide(const DrawingsTimeline& drawings, Policy policy, Pacing pacing, std::uint32_t left,
                       Ratio t) {
  auto decision = drawings.decide(policy, pacing, id(left), t);
  REQUIRE(decision);
  return *decision;
}

InstantDecision source(std::uint32_t a) {
  return {.show = Show::Source, .a = id(a)};
}
InstantDecision repeated(std::uint32_t a) {
  return {.show = Show::Repeated, .a = id(a)};
}
InstantDecision interpolate(std::uint32_t a, std::uint32_t b, Ratio u) {
  return {.show = Show::Interpolate, .a = id(a), .b = id(b), .u = u};
}

template <class T>
RefusalCode code_of(const std::expected<T, ajp::Refusal>& result) {
  REQUIRE_FALSE(result);
  return result.error().code;
}

}  // namespace

TEST_CASE("drawings : préchargement de 4 sources en Smooth, 1 en KeepPoses", "[drawings]") {
  CHECK(preload(Pacing::Source) == 0);
  CHECK(preload(Pacing::Smooth) == 4);
  CHECK(preload(Pacing::KeepPoses) == 1);
}

TEST_CASE("drawings : un dessin est tenu jusqu'à 0,70 niveau, 0,40 est du bruit", "[drawings]") {
  DrawingsTimeline drawings;
  drawings.start(id(0));
  auto noise = drawings.push(id(1), {.worst_block_diff = 0.40});
  REQUIRE(noise);
  CHECK(noise->kind == DrawingClass::Held);
  CHECK(noise->first == id(0));
  auto edge = drawings.push(id(2), {.worst_block_diff = 0.70});
  REQUIRE(edge);
  CHECK(edge->kind == DrawingClass::Held);
  auto moved = drawings.push(id(3), {.worst_block_diff = 0.71});
  REQUIRE(moved);
  CHECK(moved->kind == DrawingClass::New);
  CHECK(moved->first == id(3));
  CHECK_FALSE(moved->scene_cut);
  auto cut = drawings.push(id(4), {.worst_block_diff = 0.0, .scene_cut = true});
  REQUIRE(cut);
  CHECK(cut->kind == DrawingClass::New);
  CHECK(cut->scene_cut);
  auto exact = drawings.push(id(5), {.identical = true});
  REQUIRE(exact);
  CHECK(exact->kind == DrawingClass::Held);
  CHECK(exact->identical);
  CHECK(exact->first == id(4));
}

TEST_CASE("drawings : mesures et ordre refusés", "[drawings]") {
  DrawingsTimeline drawings;
  CHECK(code_of(drawings.push(id(1), measure('N'))) == RefusalCode::SequenceOrder);
  drawings.start(id(0));
  const double kNan = std::numeric_limits<double>::quiet_NaN();
  const double kInf = std::numeric_limits<double>::infinity();
  for (const PairMeasure bad : {PairMeasure{.worst_block_diff = kNan}, PairMeasure{.worst_block_diff = -0.1},
                                PairMeasure{.worst_block_diff = 255.5}, PairMeasure{.worst_block_diff = kInf},
                                PairMeasure{.worst_block_diff = 0.1, .identical = true},
                                PairMeasure{.scene_cut = true, .identical = true}}) {
    CHECK(code_of(drawings.push(id(1), bad)) == RefusalCode::FieldOutOfRange);
  }
  CHECK(drawings.push(id(1), {.worst_block_diff = 255.0}));
  drawings.finish();
  CHECK(code_of(drawings.push(id(2), measure('N'))) == RefusalCode::SequenceOrder);
  drawings.start(id(10));
  CHECK(drawings.push(id(11), measure('h')));
  CHECK_FALSE(drawings.ready(id(0), Pacing::Source));  // start() oublie la séquence passée
  // Une identité déjà classée rendrait la première paire introuvable.
  CHECK(code_of(drawings.push(id(10), measure('N'))) == RefusalCode::SequenceOrder);
  CHECK(code_of(drawings.push(id(11), measure('N'))) == RefusalCode::SequenceOrder);
  CHECK(drawings.push(id(12), measure('N')));
}

TEST_CASE("drawings : plans en uns, en deux, en trois", "[drawings]") {
  CHECK(timeline("N").rhythm() == Rhythm::Unknown);
  CHECK(timeline("NNNNN").rhythm() == Rhythm::Ones);
  CHECK(timeline("NhNhNhN").rhythm() == Rhythm::Twos);
  CHECK(timeline("NiNhNhN").rhythm() == Rhythm::Twos);
  CHECK(timeline("NhhNhhNhhN").rhythm() == Rhythm::Threes);
  CHECK(timeline("NhhhhNhhhhhhN").rhythm() == Rhythm::Longer);
  // Le dessin en cours ne compte pas tant qu'il n'est pas achevé.
  CHECK(timeline("NhNhNhh").rhythm() == Rhythm::Twos);
  // À égalité, la tenue la plus courte : elle demande le plus de calcul.
  CHECK(timeline("NNhNNhN").rhythm() == Rhythm::Ones);
  // Une coupe ouvre un plan : le rythme repart de zéro.
  CHECK(timeline("NhNhNC").rhythm() == Rhythm::Unknown);
  CHECK(timeline("NNNNNChNhNhN").rhythm() == Rhythm::Twos);
}

TEST_CASE("drawings : le rythme suit les 8 dernières tenues du plan", "[drawings]") {
  std::string pattern = "N";
  for (int i = 0; i < 8; ++i) pattern += "hN";
  CHECK(timeline(pattern + "NNN").rhythm() == Rhythm::Twos);    // 5 en deux, 3 en uns
  CHECK(timeline(pattern + "NNNNN").rhythm() == Rhythm::Ones);  // 3 en deux, 5 en uns
}

TEST_CASE("drawings : rythme Source, Fast reprend les dessins tenus", "[drawings]") {
  const DrawingsTimeline drawings = timeline("NhNC");
  for (Policy policy : {Policy::Fast, Policy::Refine}) {
    CHECK(decide(drawings, policy, Pacing::Source, 0, {1, 2}) == repeated(0));
    CHECK(decide(drawings, policy, Pacing::Source, 0, {2, 2}) == source(1));
    CHECK(decide(drawings, policy, Pacing::Source, 1, {1, 4}) == interpolate(1, 2, {1, 4}));
    CHECK(decide(drawings, policy, Pacing::Source, 2, {1, 2}) == repeated(2));  // coupe
    CHECK(decide(drawings, policy, Pacing::Source, 2, {2, 2}) == source(3));
  }
}

TEST_CASE("drawings : rythme Source, Quality ne reprend qu'un doublon exact", "[drawings]") {
  const DrawingsTimeline drawings = timeline("NhiNC");
  CHECK(decide(drawings, Policy::Quality, Pacing::Source, 0, {1, 2}) == interpolate(0, 1, {1, 2}));
  CHECK(decide(drawings, Policy::Quality, Pacing::Source, 1, {2, 6}) == repeated(1));
  CHECK(decide(drawings, Policy::Quality, Pacing::Source, 1, {6, 6}) == source(2));
  CHECK(decide(drawings, Policy::Quality, Pacing::Source, 2, {1, 3}) == interpolate(2, 3, {1, 3}));
  CHECK(decide(drawings, Policy::Quality, Pacing::Source, 3, {1, 3}) == repeated(3));  // coupe, comme le fork
}

TEST_CASE("drawings : KeepPoses garde la grille et précharge une source", "[drawings]") {
  DrawingsTimeline drawings = timeline("NhN", false);
  CHECK(drawings.ready(id(0), Pacing::KeepPoses));
  CHECK_FALSE(drawings.ready(id(1), Pacing::KeepPoses));
  CHECK(code_of(drawings.decide(Policy::Fast, Pacing::KeepPoses, id(1), {1, 2})) ==
        RefusalCode::LookaheadMissing);
  CHECK(decide(drawings, Policy::Fast, Pacing::KeepPoses, 0, {1, 2}) == repeated(0));
  REQUIRE(drawings.push(id(3), measure('N')));
  CHECK(decide(drawings, Policy::Fast, Pacing::KeepPoses, 1, {1, 2}) == interpolate(1, 2, {1, 2}));
}

TEST_CASE("drawings : Smooth répartit un plan en deux sur toute la tenue", "[drawings]") {
  const DrawingsTimeline drawings = timeline("NhNhN");
  CHECK(decide(drawings, Policy::Fast, Pacing::Smooth, 0, {1, 2}) == interpolate(0, 2, {1, 4}));
  CHECK(decide(drawings, Policy::Fast, Pacing::Smooth, 0, {2, 2}) == interpolate(0, 2, {1, 2}));
  CHECK(decide(drawings, Policy::Fast, Pacing::Smooth, 1, {1, 2}) == interpolate(0, 2, {3, 4}));
  CHECK(decide(drawings, Policy::Fast, Pacing::Smooth, 1, {2, 2}) == source(2));
  CHECK(decide(drawings, Policy::Refine, Pacing::Smooth, 2, {1, 2}) == interpolate(2, 4, {1, 4}));
}

TEST_CASE("drawings : Smooth suit les dessins pour chaque tenue et chaque facteur", "[drawings]") {
  for (std::uint32_t hold = 1; hold <= 4; ++hold) {
    for (const Ratio factor :
         {Ratio{2, 1}, Ratio{3, 1}, Ratio{4, 1}, Ratio{5, 2}, Ratio{3, 2}, Ratio{11011, 3200}}) {
      CAPTURE(hold, factor.num, factor.den);
      std::string pattern;
      for (int drawing = 0; drawing < 6; ++drawing) pattern += "N" + std::string(hold - 1, 'h');
      pattern += "N";
      const DrawingsTimeline drawings = timeline(pattern);
      auto cadence = CadenceAccumulator::create(factor);
      REQUIRE(cadence);
      bool all_on_drawings = true;
      for (std::uint32_t k = 0; k < pattern.size(); ++k) {
        const CadenceStep step = cadence->push({.id = id(k), .pts_us = std::int64_t{k} * 40000});
        if (step.event != GridEvent::Pair) continue;
        const std::uint32_t left = step.left.value - kFirst;
        const std::uint32_t first = left / hold * hold;  // première source du dessin à l'écran
        for (const OutputInstant& instant : step.instants()) {
          const InstantDecision d = decide(drawings, Policy::Fast, Pacing::Smooth, left, instant.t);
          // La position exacte de l'instant, en sources, dans la tenue du dessin : (left + t - first) / hold.
          const std::int64_t num = (std::int64_t{left} - first) * instant.t.den + instant.t.num;
          const std::int64_t den = std::int64_t{hold} * instant.t.den;
          if (num == den) {
            all_on_drawings = all_on_drawings && d == source(first + hold);
          } else {
            all_on_drawings = all_on_drawings && d.show == Show::Interpolate && d.a == id(first) &&
                              d.b == id(first + hold) && d.u.num * den == num * d.u.den;
          }
        }
      }
      CHECK(all_on_drawings);
    }
  }
}

TEST_CASE("drawings : Smooth suit des tenues inégales, dessin par dessin", "[drawings]") {
  // Dessins : 0-1 (tenue 2), 2-4 (3), 5 (1), 6-9 (4), puis 10.
  const DrawingsTimeline drawings = timeline("NhNhhNNhhhN");
  constexpr Ratio kHalf{1, 2};
  CHECK(decide(drawings, Policy::Fast, Pacing::Smooth, 0, kHalf) == interpolate(0, 2, {1, 4}));
  CHECK(decide(drawings, Policy::Fast, Pacing::Smooth, 1, kHalf) == interpolate(0, 2, {3, 4}));
  CHECK(decide(drawings, Policy::Fast, Pacing::Smooth, 1, {1, 1}) == source(2));
  CHECK(decide(drawings, Policy::Fast, Pacing::Smooth, 2, kHalf) == interpolate(2, 5, {1, 6}));
  CHECK(decide(drawings, Policy::Fast, Pacing::Smooth, 3, kHalf) == interpolate(2, 5, {1, 2}));
  CHECK(decide(drawings, Policy::Fast, Pacing::Smooth, 3, {1, 1}) == interpolate(2, 5, {2, 3}));
  CHECK(decide(drawings, Policy::Fast, Pacing::Smooth, 4, kHalf) == interpolate(2, 5, {5, 6}));
  CHECK(decide(drawings, Policy::Fast, Pacing::Smooth, 4, {1, 1}) == source(5));
  CHECK(decide(drawings, Policy::Fast, Pacing::Smooth, 5, kHalf) == interpolate(5, 6, {1, 2}));
  CHECK(decide(drawings, Policy::Fast, Pacing::Smooth, 6, kHalf) == interpolate(6, 10, {1, 8}));
  CHECK(decide(drawings, Policy::Fast, Pacing::Smooth, 8, {1, 3}) == interpolate(6, 10, {7, 12}));
  CHECK(decide(drawings, Policy::Fast, Pacing::Smooth, 9, kHalf) == interpolate(6, 10, {7, 8}));
  CHECK(decide(drawings, Policy::Fast, Pacing::Smooth, 9, {1, 1}) == source(10));

  // Le même plan au rythme Source : le mouvement tient dans la paire qui change de dessin.
  CHECK(decide(drawings, Policy::Fast, Pacing::Source, 3, kHalf) == repeated(3));
  CHECK(decide(drawings, Policy::Fast, Pacing::Source, 4, kHalf) == interpolate(4, 5, kHalf));
  CHECK(decide(drawings, Policy::Quality, Pacing::Source, 3, kHalf) == interpolate(3, 4, kHalf));
}

TEST_CASE("drawings : Smooth montre un plan fixe tel quel, au-delà de 4 sources", "[drawings]") {
  const DrawingsTimeline still = timeline("NhhhhhhN");
  CHECK(decide(still, Policy::Fast, Pacing::Smooth, 0, {1, 2}) == repeated(0));
  CHECK(decide(still, Policy::Fast, Pacing::Smooth, 3, {2, 2}) == repeated(0));
  // Le plan fixe s'achève : le mouvement tient dans la dernière paire.
  CHECK(decide(still, Policy::Fast, Pacing::Smooth, 6, {1, 2}) == interpolate(0, 7, {1, 2}));
  CHECK(decide(still, Policy::Fast, Pacing::Smooth, 6, {2, 2}) == source(7));

  const DrawingsTimeline four = timeline("NhhhN");
  CHECK(decide(four, Policy::Fast, Pacing::Smooth, 0, {1, 2}) == interpolate(0, 4, {1, 8}));
  const DrawingsTimeline five = timeline("NhhhhN");
  CHECK(decide(five, Policy::Fast, Pacing::Smooth, 0, {1, 2}) == repeated(0));
  CHECK(decide(five, Policy::Fast, Pacing::Smooth, 4, {1, 2}) == interpolate(0, 5, {1, 2}));
}

TEST_CASE("drawings : Smooth n'interpole jamais à travers une coupe", "[drawings]") {
  const DrawingsTimeline drawings = timeline("NhChN");
  CHECK(decide(drawings, Policy::Fast, Pacing::Smooth, 0, {1, 2}) == repeated(0));
  CHECK(decide(drawings, Policy::Fast, Pacing::Smooth, 1, {1, 2}) == repeated(0));
  CHECK(decide(drawings, Policy::Fast, Pacing::Smooth, 1, {2, 2}) == source(2));
  CHECK(decide(drawings, Policy::Fast, Pacing::Smooth, 2, {1, 2}) == interpolate(2, 4, {1, 4}));
}

TEST_CASE("drawings : Smooth attend son préchargement, ou la fin du flux", "[drawings]") {
  DrawingsTimeline drawings = timeline("Nhhh", false);
  CHECK_FALSE(drawings.ready(id(0), Pacing::Smooth));
  CHECK(drawings.ready(id(0), Pacing::Source));
  CHECK(code_of(drawings.decide(Policy::Fast, Pacing::Smooth, id(0), {1, 2})) ==
        RefusalCode::LookaheadMissing);
  REQUIRE(drawings.push(id(4), measure('N')));
  REQUIRE(drawings.push(id(5), measure('h')));
  CHECK(drawings.ready(id(0), Pacing::Smooth));
  CHECK(decide(drawings, Policy::Fast, Pacing::Smooth, 0, {1, 2}) == interpolate(0, 4, {1, 8}));

  // À la fin du flux, l'anticipation qui manque vaut « aucun dessin suivant ».
  DrawingsTimeline ending = timeline("Nhh", false);
  ending.finish();
  CHECK(ending.ready(id(0), Pacing::Smooth));
  CHECK(decide(ending, Policy::Fast, Pacing::Smooth, 1, {1, 2}) == repeated(0));
}

TEST_CASE("drawings : décisions refusées", "[drawings]") {
  const DrawingsTimeline drawings = timeline("NhN");
  CHECK(code_of(drawings.decide(Policy::Quality, Pacing::Smooth, id(0), {1, 2})) ==
        RefusalCode::PacingForbidden);
  CHECK(code_of(drawings.decide(Policy::Quality, Pacing::KeepPoses, id(0), {1, 2})) ==
        RefusalCode::PacingForbidden);
  constexpr std::int64_t kHuge = std::numeric_limits<std::int64_t>::max();
  for (const Ratio bad :
       {Ratio{0, 1}, Ratio{3, 2}, Ratio{1, 0}, Ratio{-1, 2}, Ratio{1, kHuge}, Ratio{1, 100001}}) {
    CHECK(code_of(drawings.decide(Policy::Fast, Pacing::Source, id(0), bad)) == RefusalCode::FieldOutOfRange);
    CHECK(code_of(drawings.decide(Policy::Fast, Pacing::Smooth, id(0), bad)) == RefusalCode::FieldOutOfRange);
  }
  CHECK(drawings.decide(Policy::Fast, Pacing::Smooth, id(0), {1, 100000}));
  CHECK(code_of(drawings.decide(Policy::Fast, Pacing::Source, id(9), {1, 2})) == RefusalCode::SourceUnknown);
  // La dernière source n'ouvre aucune paire.
  CHECK(code_of(drawings.decide(Policy::Fast, Pacing::Source, id(2), {1, 2})) == RefusalCode::SourceUnknown);

  // Une source sortie de la fenêtre est oubliée.
  const DrawingsTimeline long_run = timeline(std::string(drawings::kWindow + 4, 'N'));
  CHECK(code_of(long_run.decide(Policy::Fast, Pacing::Source, id(0), {1, 2})) == RefusalCode::SourceUnknown);
  CHECK(long_run.decide(Policy::Fast, Pacing::Source, id(drawings::kWindow), {1, 2}));
}

TEST_CASE("drawings : en Smooth, l'image montrée avance avec le temps, jamais à travers une coupe",
          "[drawings]") {
  // Motifs pseudo-aléatoires fixés ; la position montrée, en sources, est a pour
  // Source et Repeated, a + u·(b - a) pour Interpolate.
  std::mt19937 random(1707);  // mt19937 est défini par la norme : la suite est la même partout
  constexpr std::array kLetters{'N', 'N', 'N', 'N', 'h', 'h', 'h', 'h', 'i', 'C'};
  constexpr std::array kFactors{Ratio{2, 1}, Ratio{4, 1}, Ratio{5, 2}, Ratio{3, 2}, Ratio{11011, 3200}};
  for (int run = 0; run < 500; ++run) {
    const std::size_t length = 12 + random() % 20;  // la fenêtre de 32 sources garde tout le motif
    std::string pattern = "N";
    while (pattern.size() < length) pattern += kLetters[random() % kLetters.size()];
    const Ratio factor = kFactors[random() % kFactors.size()];
    CAPTURE(pattern, factor.num, factor.den);
    const DrawingsTimeline drawings = timeline(pattern);
    auto cadence = CadenceAccumulator::create(factor);
    REQUIRE(cadence);

    bool forward = true;  // jamais en arrière, jamais en avance sur l'instant
    bool exact = true;    // une tenue répartie montre l'instant lui-même
    bool no_cut = true;
    double shown = 0.0;
    for (std::uint32_t k = 0; k < pattern.size(); ++k) {
      const CadenceStep step = cadence->push({.id = id(k), .pts_us = std::int64_t{k} * 41708});
      if (step.event != GridEvent::Pair) continue;
      const std::uint32_t left = step.left.value - kFirst;
      for (const OutputInstant& instant : step.instants()) {
        const InstantDecision d = decide(drawings, Policy::Fast, Pacing::Smooth, left, instant.t);
        const double at = left + instant.t.value();
        const double a = d.a.value - kFirst;
        double position = a;
        if (d.show == Show::Interpolate) {
          const double b = d.b.value - kFirst;
          position = a + d.u.value() * (b - a);
          exact = exact && (b - a > drawings::kMaxSpreadHold || std::abs(position - at) < 1e-9);
          for (auto c = static_cast<std::size_t>(a) + 1; c <= static_cast<std::size_t>(b); ++c) {
            no_cut = no_cut && pattern[c] != 'C';
          }
        }
        if (d.show == Show::Source) exact = exact && position == at;
        forward = forward && position >= shown - 1e-9 && position <= at + 1e-9;
        shown = position;
      }
    }
    CHECK(forward);
    CHECK(exact);
    CHECK(no_cut);
  }
}
