// Dépendances entre voies d'un backend (gpu/lanes.h), sans GPU : quelles attentes une opération
// pose avant elle, selon ce qu'elle lit et écrit et ce que les autres voies ont fait.
#include "gpu/lanes.h"

#include <array>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace ajp::gpu {
namespace {

using Waits = std::vector<LaneHazards::Wait>;

constexpr std::uintptr_t kFrame = 0x1000;  // image ingérée
constexpr std::uintptr_t kSpan = 0x2000;   // sortie de SPAN
constexpr std::uintptr_t kFlow = 0x3000;   // flot de RIFE
constexpr std::uintptr_t kNext = 0x4000;   // image suivante

LaneHazards::Op op(LaneHazards& hazards, Lane lane, std::initializer_list<std::uintptr_t> reads,
                   std::initializer_list<std::uintptr_t> writes) {
  return hazards.access(lane, std::span(reads.begin(), reads.size()),
                        std::span(writes.begin(), writes.size()));
}

}  // namespace

TEST_CASE("Voies : sur une même voie, l'ordre de la file suffit", "[gpu][lanes]") {
  LaneHazards hazards;
  CHECK(op(hazards, Lane::Upscale, {}, {kFrame}).waits.empty());
  const LaneHazards::Op span = op(hazards, Lane::Upscale, {kFrame}, {kSpan});
  CHECK(span.mark == 2);
  CHECK(span.waits.empty());
  CHECK(op(hazards, Lane::Upscale, {kSpan}, {kFrame}).waits.empty());
  CHECK(hazards.last(Lane::Upscale) == 3);
  CHECK(hazards.last(Lane::Measure) == 0);
}

TEST_CASE("Voies : lire ce qu'une autre voie a écrit attend cette écriture, une seule fois", "[gpu][lanes]") {
  LaneHazards hazards;
  CHECK(op(hazards, Lane::Measure, {}, {kFrame}).mark == 1);
  const LaneHazards::Op span = op(hazards, Lane::Upscale, {kFrame}, {kSpan});
  CHECK(span.waits == Waits{{.lane = Lane::Measure, .mark = 1}});
  // Déjà attendue par la file de l'agrandissement : pas répétée.
  CHECK(op(hazards, Lane::Upscale, {kFrame}, {kSpan}).waits.empty());
  // Une autre voie, elle, l'attend à son tour.
  CHECK(op(hazards, Lane::Interpolate, {kFrame}, {kFlow}).waits == Waits{{.lane = Lane::Measure, .mark = 1}});
}

TEST_CASE("Voies : écrire ce qu'une autre voie lit encore attend sa lecture", "[gpu][lanes]") {
  LaneHazards hazards;
  op(hazards, Lane::Measure, {}, {kFrame});                               // Measure 1
  op(hazards, Lane::Upscale, {kFrame}, {kSpan});                          // Upscale 1 lit l'image
  const LaneHazards::Op next = op(hazards, Lane::Measure, {}, {kFrame});  // réécrite (Measure 2)
  CHECK(next.waits == Waits{{.lane = Lane::Upscale, .mark = 1}});
  // Écriture après écriture d'une autre voie : attendue aussi.
  CHECK(op(hazards, Lane::Interpolate, {}, {kSpan}).waits == Waits{{.lane = Lane::Upscale, .mark = 1}});
}

TEST_CASE("Voies : une attente par autre voie, la plus récente", "[gpu][lanes]") {
  LaneHazards hazards;
  op(hazards, Lane::Measure, {}, {kFrame});  // Measure 1
  op(hazards, Lane::Measure, {}, {kNext});   // Measure 2
  op(hazards, Lane::Upscale, {}, {kSpan});   // Upscale 1
  const LaneHazards::Op warp = op(hazards, Lane::Interpolate, {kFrame, kNext, kSpan}, {kFlow});
  CHECK(warp.waits == Waits{{.lane = Lane::Measure, .mark = 2}, {.lane = Lane::Upscale, .mark = 1}});
  // Une ressource lue et écrite sur place par une autre voie : son écriture et ses lectures.
  const LaneHazards::Op in_place = op(hazards, Lane::Upscale, {kFlow}, {kFlow});
  CHECK(in_place.waits == Waits{{.lane = Lane::Interpolate, .mark = 1}});
}

TEST_CASE("Voies : la jointure suit le dernier travail de chaque autre voie", "[gpu][lanes]") {
  LaneHazards hazards;
  CHECK(hazards.join(Lane::Upscale).waits.empty());  // rien ailleurs
  op(hazards, Lane::Measure, {}, {kFrame});
  op(hazards, Lane::Interpolate, {}, {kFlow});
  op(hazards, Lane::Interpolate, {}, {kFlow});
  const LaneHazards::Op joined = hazards.join(Lane::Upscale);
  CHECK(joined.waits == Waits{{.lane = Lane::Measure, .mark = 1}, {.lane = Lane::Interpolate, .mark = 2}});
  CHECK(hazards.join(Lane::Upscale).waits.empty());  // déjà suivi
  // Après la jointure, lire ce que les autres voies avaient écrit n'attend plus rien.
  CHECK(op(hazards, Lane::Upscale, {kFrame, kFlow}, {kSpan}).waits.empty());
}

}  // namespace ajp::gpu
