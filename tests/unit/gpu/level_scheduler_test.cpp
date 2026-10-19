// Ordonnanceur d'un niveau : instants de la cadence, décisions de la timeline des dessins,
// identités dérivées, anticipation et entrées gardées vivantes.
#include "gpu/level_scheduler.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using ajp::RefusalCode;
using ajp::SourceId;
using namespace ajp::gpu;
namespace core = ajp::core;

namespace {

constexpr std::int64_t kFrameUs = 40'000;

core::InterpLevel level(core::Ratio factor, core::Policy policy = core::Policy::Quality,
                        core::Pacing pacing = core::Pacing::Source) {
  return {.factor = factor, .model = "rife", .policy = policy, .pacing = pacing};
}

// Une entrée et sa mesure contre la précédente : nouveau dessin, tenu, doublon ou coupe.
enum class Kind { New, Held, Identical, Cut };

LevelInput input(std::uint32_t id, Kind kind = Kind::New, std::uint32_t epoch = 0) {
  core::PairMeasure measure{.worst_block_diff = 10.0};
  if (kind == Kind::Held) measure = {.worst_block_diff = 0.5};
  if (kind == Kind::Identical) measure = {.worst_block_diff = 0.0, .identical = true};
  if (kind == Kind::Cut) measure = {.worst_block_diff = 80.0, .scene_cut = true};
  return {.id = SourceId{id}, .pts_us = std::int64_t{id} * kFrameUs, .epoch = epoch, .measure = measure};
}

std::vector<LevelOutput> push(LevelScheduler& scheduler, const LevelInput& in) {
  auto outputs = scheduler.push(in);
  REQUIRE(outputs);
  return {outputs->begin(), outputs->end()};
}

std::vector<LevelOutput> push_all(LevelScheduler& scheduler, const std::vector<LevelInput>& inputs) {
  std::vector<LevelOutput> all;
  for (const LevelInput& in : inputs) {
    const auto outputs = push(scheduler, in);
    all.insert(all.end(), outputs.begin(), outputs.end());
  }
  return all;
}

std::vector<std::int64_t> pts_of(const std::vector<LevelOutput>& outputs) {
  std::vector<std::int64_t> pts;
  for (const LevelOutput& output : outputs) pts.push_back(output.pts_us.value_or(-1));
  return pts;
}

}  // namespace

TEST_CASE("Niveau : x2 en Quality émet la source puis le milieu de chaque paire", "[level]") {
  auto scheduler = LevelScheduler::create(level({2, 1}), 0);
  REQUIRE(scheduler);
  const auto outputs = push_all(*scheduler, {input(0), input(1), input(2), input(3)});

  CHECK(pts_of(outputs) == std::vector<std::int64_t>{0, 20'000, 40'000, 60'000, 80'000, 100'000, 120'000});
  CHECK(outputs[0].decision == core::InstantDecision{.show = core::Show::Source, .a = SourceId{0}});
  CHECK(outputs[1].decision ==
        core::InstantDecision{
            .show = core::Show::Interpolate, .a = SourceId{0}, .b = SourceId{1}, .u = {1, 2}});
  CHECK(outputs[2].decision == core::InstantDecision{.show = core::Show::Source, .a = SourceId{1}});
  CHECK(outputs[1].pair_left == SourceId{0});
  CHECK(outputs[1].pair_right == SourceId{1});

  std::set<std::uint32_t> ids;
  for (const LevelOutput& output : outputs) {
    CHECK(is_derived(output.id));
    ids.insert(output.id.value);
  }
  CHECK(ids.size() == outputs.size());  // chaque sortie a sa propre identité
  CHECK(derived_id(1, 0) != derived_id(0, 0));
  CHECK(scheduler->counters().interpolated == 3);
  CHECK(scheduler->counters().outputs == 7);
}

TEST_CASE("Niveau : 5/2 et x3 suivent la grille de l'accumulateur", "[level]") {
  SECTION("5/2 : deux instants puis trois, sur une grille de 16 ms") {
    auto scheduler = LevelScheduler::create(level({5, 2}), 0);
    REQUIRE(scheduler);
    const auto outputs = push_all(*scheduler, {input(0), input(1), input(2), input(3), input(4)});
    REQUIRE(outputs.size() == 11);
    for (std::size_t i = 0; i < outputs.size(); ++i) CHECK(outputs[i].pts_us == std::int64_t(i) * 16'000);
    CHECK(outputs[5].decision == core::InstantDecision{.show = core::Show::Source, .a = SourceId{2}});
  }
  SECTION("x3 : deux instants calculés et la source par paire") {
    auto scheduler = LevelScheduler::create(level({3, 1}), 0);
    REQUIRE(scheduler);
    const auto outputs = push_all(*scheduler, {input(0), input(1), input(2)});
    REQUIRE(outputs.size() == 7);
    CHECK(outputs[1].decision.u == core::Ratio{1, 3});
    CHECK(outputs[2].decision.u == core::Ratio{2, 3});
    CHECK(std::ranges::is_sorted(pts_of(outputs)));
    CHECK(scheduler->counters().interpolated == 4);
  }
}

TEST_CASE("Niveau : dessins tenus, doublons et coupes", "[level]") {
  SECTION("Fast reprend un dessin tenu (REPEATED), Quality l'interpole") {
    auto fast = LevelScheduler::create(level({2, 1}, core::Policy::Fast), 0);
    auto quality = LevelScheduler::create(level({2, 1}), 0);
    REQUIRE(fast);
    REQUIRE(quality);
    const std::vector inputs{input(0), input(1, Kind::Held), input(2)};
    const auto f = push_all(*fast, inputs);
    const auto q = push_all(*quality, inputs);
    CHECK(f[1].decision == core::InstantDecision{.show = core::Show::Repeated, .a = SourceId{0}});
    CHECK(f[3].decision.show == core::Show::Interpolate);
    CHECK(q[1].decision.show == core::Show::Interpolate);
    CHECK(fast->counters().repeated == 1);
  }
  SECTION("Quality ne reprend qu'un doublon exact, et le dit au niveau suivant") {
    auto scheduler = LevelScheduler::create(level({2, 1}), 0);
    REQUIRE(scheduler);
    const auto outputs = push_all(*scheduler, {input(0), input(1, Kind::Identical), input(2)});
    CHECK(outputs[1].decision.show == core::Show::Repeated);
    CHECK(outputs[1].measure.identical);  // Repeated(0) après Source(0)
    CHECK(outputs[2].measure.identical);  // Source(1), doublon exact de 0
    CHECK_FALSE(outputs[3].measure.identical);
  }
  SECTION("une coupe n'est jamais interpolée : l'image de gauche reste") {
    auto scheduler = LevelScheduler::create(level({2, 1}, core::Policy::Fast), 0);
    REQUIRE(scheduler);
    const auto outputs = push_all(*scheduler, {input(0), input(1), input(2, Kind::Cut)});
    CHECK(outputs[3].decision == core::InstantDecision{.show = core::Show::Repeated, .a = SourceId{1}});
    CHECK(outputs[3].scene_cut);
    CHECK_FALSE(outputs[4].scene_cut);    // la source droite elle-même
    CHECK(outputs[4].measure.scene_cut);  // le niveau suivant voit la coupe
    CHECK(scheduler->counters().scene_cuts == 1);
  }
}

TEST_CASE("Niveau : Flush, époques et remise à zéro", "[level]") {
  auto scheduler = LevelScheduler::create(level({2, 1}), 0);
  REQUIRE(scheduler);
  push_all(*scheduler, {input(0), input(1)});
  REQUIRE(scheduler->flush(1));
  CHECK(push(*scheduler, input(2, Kind::New, 0)).empty());  // époque passée : jetée
  const auto restart = push(*scheduler, input(3, Kind::New, 1));
  REQUIRE(restart.size() == 1);
  CHECK(restart[0].decision == core::InstantDecision{.show = core::Show::Source, .a = SourceId{3}});
  CHECK(scheduler->flush(0).error().code == RefusalCode::SequenceOrder);

  // Une identité reprise est refusée ; le niveau repart de l'entrée suivante sans reculer
  // d'époque (l'accumulateur jetterait sinon toutes les entrées de l'époque 1).
  auto refused = scheduler->push(input(3, Kind::New, 1));
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::SequenceOrder);
  CHECK(scheduler->live().empty());
  const auto after = push(*scheduler, input(4, Kind::New, 1));
  REQUIRE(after.size() == 1);
  CHECK(after[0].decision.a == SourceId{4});
  CHECK(push(*scheduler, input(5, Kind::New, 1)).size() == 2);
}

TEST_CASE("Niveau : Smooth attend son anticipation et garde ce qu'il citera", "[level]") {
  auto scheduler = LevelScheduler::create(level({2, 1}, core::Policy::Fast, core::Pacing::Smooth), 0);
  REQUIRE(scheduler);
  // En deux : chaque dessin tient deux sources.
  std::vector<LevelOutput> all = push(*scheduler, input(0));
  for (std::uint32_t id = 1; id < 5; ++id) {
    const auto outputs = push(*scheduler, input(id, id % 2 == 1 ? Kind::Held : Kind::New));
    all.insert(all.end(), outputs.begin(), outputs.end());
  }
  CHECK(all.size() == 1);  // la première paire attend 4 sources classées au-delà de sa droite
  const auto more = push(*scheduler, input(5, Kind::Held));
  REQUIRE_FALSE(more.empty());
  // La paire (0, 1) tient le dessin 0 : son milieu interpole vers le dessin suivant (2).
  CHECK(more[0].decision.show == core::Show::Interpolate);
  CHECK(more[0].decision.b == SourceId{2});
  const auto live = scheduler->live();
  CHECK(std::ranges::find(live, SourceId{5}) != live.end());
  CHECK(scheduler->finish().size() > 0);  // la fin du flux libère les paires restantes
}

TEST_CASE("Niveau : toute décision cite une entrée encore vivante, dans l'ordre des pts", "[level]") {
  const std::vector<core::InterpLevel> levels{
      level({2, 1}),
      level({5, 2}, core::Policy::Fast),
      level({2, 1}, core::Policy::Fast, core::Pacing::Smooth),
      level({3, 1}, core::Policy::Refine, core::Pacing::KeepPoses),
  };
  for (const core::InterpLevel& l : levels) {
    auto scheduler = LevelScheduler::create(l, 1);
    REQUIRE(scheduler);
    std::mt19937 random(7);
    std::vector<SourceId> alive;
    std::int64_t last_pts = -1;
    for (std::uint32_t id = 0; id < 300; ++id) {
      const Kind kind =
          id == 0 ? Kind::New
                  : std::array{Kind::New, Kind::Held, Kind::Held, Kind::Identical, Kind::Cut}[random() % 5];
      alive.assign(scheduler->live().begin(), scheduler->live().end());
      alive.push_back(SourceId{id});
      for (const LevelOutput& output : push(*scheduler, input(id, kind))) {
        CHECK(std::ranges::find(alive, output.decision.a) != alive.end());
        if (output.decision.show == core::Show::Interpolate) {
          CHECK(std::ranges::find(alive, output.decision.b) != alive.end());
        }
        CHECK(output.pts_us.value_or(-1) > last_pts);
        last_pts = output.pts_us.value_or(-1);
        CHECK(((output.id.value >> 28) & 0x7U) == 1);
      }
    }
  }
}

TEST_CASE("Niveau : refus à la création", "[level]") {
  CHECK(LevelScheduler::create(level({2, 1}), kMaxLevels).error().code == RefusalCode::FieldOutOfRange);
  CHECK(LevelScheduler::create(level({2, 1}, core::Policy::Quality, core::Pacing::Smooth), 0).error().code ==
        RefusalCode::PacingForbidden);
  CHECK_FALSE(LevelScheduler::create(level({1, 1}), 0));
  CHECK_FALSE(LevelScheduler::create(level({9, 1}), 0));
}
