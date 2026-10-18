#include "core/pacing.h"

#include <cmath>
#include <cstdint>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/cadence.h"

using ajp::RefusalCode;
using Catch::Approx;
using namespace ajp::core;

namespace {

constexpr double kNtsc = 24000.0 / 1001.0;  // 23,976 im/s

DisplayInfo fixed(double hz) {
  return {.width = 3840, .height = 2160, .refresh_hz = hz};
}

DisplayInfo vrr(double hz) {
  return {.width = 3840, .height = 2160, .refresh_hz = hz, .vrr = true};
}

CadenceChoice choose(const DisplayInfo& display, Ratio requested, double max_factor = 8.0,
                     double fps = kNtsc) {
  auto choice = choose_cadence(display, fps, requested, max_factor);
  REQUIRE(choice);
  return *choice;
}

Step interpolate(Ratio factor, Policy policy = Policy::Fast) {
  return {.kind = StepKind::Interpolate, .model = "rife", .factor = factor, .policy = policy};
}

template <class T>
RefusalCode code_of(const std::expected<T, ajp::Refusal>& result) {
  REQUIRE_FALSE(result);
  return result.error().code;
}

Step upscale(std::int64_t factor) {
  return {.kind = StepKind::Upscale, .model = "span", .factor = {factor, 1}};
}

}  // namespace

TEST_CASE("pacing : VRR garde le facteur, chaque image à son horodatage", "[pacing]") {
  const CadenceChoice choice = choose(vrr(165.0), {4, 1});
  CHECK(choice.factor == Ratio{4, 1});
  CHECK(choice.grid == Grid::Timestamps);
  CHECK(choice.speed == 1.0);
  CHECK(choice.output_fps == Approx(95.904).epsilon(1e-4));
  // Le facteur n'est pas réécrit, même non réduit.
  CHECK(choose(vrr(165.0), {6, 4}).factor == Ratio{6, 4});
}

TEST_CASE("pacing : 120 Hz fixe donne x5 ou 5/3 à la vitesse 1,001", "[pacing]") {
  const CadenceChoice x5 = choose(fixed(120.0), {4, 1});
  CHECK(x5.factor == Ratio{5, 1});
  CHECK(x5.grid == Grid::Vsync);
  CHECK(x5.vsyncs_per_frame == 1);
  CHECK(x5.output_fps == Approx(119.88).epsilon(1e-5));
  CHECK(x5.speed == Approx(1.001).epsilon(1e-9));

  const CadenceChoice five_thirds = choose(fixed(120.0), {2, 1});
  CHECK(five_thirds.factor == Ratio{5, 3});
  CHECK(five_thirds.vsyncs_per_frame == 3);
  CHECK(five_thirds.output_fps == Approx(39.96).epsilon(1e-5));
  CHECK(five_thirds.speed == Approx(1.001).epsilon(1e-9));
}

TEST_CASE("pacing : 144 Hz fixe divise le rafraîchissement, x4 exclu", "[pacing]") {
  const CadenceChoice x3 = choose(fixed(144.0), {4, 1});
  CHECK(x3.factor == Ratio{3, 1});
  CHECK(x3.vsyncs_per_frame == 2);
  CHECK(x3.output_fps == Approx(71.93).epsilon(1e-4));

  const CadenceChoice x6 = choose(fixed(144.0), {6, 1});
  CHECK(x6.factor == Ratio{6, 1});
  CHECK(x6.vsyncs_per_frame == 1);
  CHECK(x6.output_fps == Approx(143.86).epsilon(1e-4));

  // x4 imposé : 1,5 vsync par image, le motif 1-2 ne se cale pas.
  auto imposed = fit_cadence(fixed(144.0), kNtsc, {4, 1});
  REQUIRE(imposed);
  CHECK(imposed->grid == Grid::Timestamps);
  CHECK(imposed->factor == Ratio{4, 1});
}

TEST_CASE("pacing : 165 Hz fixe cale un facteur rationnel sur k vsyncs", "[pacing]") {
  const CadenceChoice k2 = choose(fixed(165.0), {4, 1});
  CHECK(k2.factor == Ratio{11011, 3200});
  CHECK(k2.vsyncs_per_frame == 2);
  CHECK(k2.output_fps == Approx(82.5).epsilon(1e-12));
  CHECK(k2.speed == Approx(1.0).epsilon(1e-12));

  const CadenceChoice k3 = choose(fixed(165.0), {2, 1});
  CHECK(k3.factor == Ratio{11011, 4800});
  CHECK(k3.vsyncs_per_frame == 3);
  CHECK(k3.output_fps == Approx(55.0).epsilon(1e-12));

  const CadenceChoice k1 = choose(fixed(165.0), {7, 1});
  CHECK(k1.factor == Ratio{11011, 1600});
  CHECK(k1.vsyncs_per_frame == 1);
}

TEST_CASE("pacing : 60 Hz fixe donne 5/2 à la vitesse 1,001", "[pacing]") {
  for (Ratio requested : {Ratio{2, 1}, Ratio{4, 1}}) {
    const CadenceChoice choice = choose(fixed(60.0), requested);
    CHECK(choice.factor == Ratio{5, 2});
    CHECK(choice.vsyncs_per_frame == 1);
    CHECK(choice.output_fps == Approx(59.94).epsilon(1e-5));
    CHECK(choice.speed == Approx(1.001).epsilon(1e-9));
  }
}

TEST_CASE("pacing : une source entière se cale sans changer de vitesse", "[pacing]") {
  const CadenceChoice choice = choose(fixed(120.0), {4, 1}, 8.0, 24.0);
  CHECK(choice.factor == Ratio{5, 1});
  CHECK(choice.speed == 1.0);
}

TEST_CASE("pacing : sans petite fraction, la fraction exacte", "[pacing]") {
  // 25 im/s sur 60 Hz : 2,4 vsyncs par source, aucune fraction de dénominateur <= 4.
  const CadenceChoice choice = choose(fixed(60.0), {2, 1}, 8.0, 25.0);
  CHECK(choice.factor == Ratio{12, 5});
  CHECK(choice.speed == Approx(1.0).epsilon(1e-12));
}

TEST_CASE("pacing : un rafraîchissement mesuré reste calé à 1e-5 près", "[pacing]") {
  const CadenceChoice choice = choose(fixed(164.9987), {4, 1});
  CHECK(choice.vsyncs_per_frame == 2);
  CHECK(std::abs(choice.speed - 1.0) <= pacing::kExactTolerance);
  CHECK(limits::kRatioTerm.contains(choice.factor.num));
  CHECK(limits::kRatioTerm.contains(choice.factor.den));
}

TEST_CASE("pacing : la borne du produit des facteurs écarte les candidats", "[pacing]") {
  CHECK(choose(fixed(165.0), {3, 1}, 3.0).factor == Ratio{11011, 4800});
  CHECK(choose(fixed(165.0), {3, 2}, 1.5).factor == Ratio{11011, 8000});  // k = 5 : 33 im/s
}

TEST_CASE("pacing : une source déjà calée n'a rien à interpoler qui se cale", "[pacing]") {
  // 59,94 im/s sur 60 Hz : la source tient 1 vsync à la vitesse 1,001 ; x2 ne se cale pas.
  const CadenceChoice none = choose(fixed(60.0), {2, 1}, 8.0, 60000.0 / 1001.0);
  CHECK(none.factor == Ratio{2, 1});
  CHECK(none.grid == Grid::Timestamps);
  // 23,976 im/s sur 120 Hz : k = 5 cale la source seule, aucun facteur 1001/1000 proposé.
  CHECK(choose(fixed(120.0), {5, 4}).factor == Ratio{5, 4});
}

TEST_CASE("pacing : chaque facteur proposé se cale et passe dans l'accumulateur", "[pacing]") {
  // Écrans et sources courants : sur la grille des vsyncs, k vsyncs par image de
  // sortie, à la vitesse proposée, font exactement le rafraîchissement.
  for (const double hz : {60.0, 60000.0 / 1001.0, 75.0, 120.0, 144.0, 165.0, 240.0}) {
    for (const double fps : {kNtsc, 24.0, 25.0, 30000.0 / 1001.0, 30.0, 50.0}) {
      for (const Ratio requested :
           {Ratio{3, 2}, Ratio{2, 1}, Ratio{5, 2}, Ratio{3, 1}, Ratio{4, 1}, Ratio{8, 1}}) {
        CAPTURE(hz, fps, requested.num, requested.den);
        const CadenceChoice choice = choose(fixed(hz), requested, 8.0, fps);
        CHECK(CadenceAccumulator::create(choice.factor));
        CHECK(choice.output_fps == Approx(fps * choice.factor.value()).epsilon(1e-12));
        if (choice.grid != Grid::Vsync) continue;
        CHECK(choice.factor == reduced(choice.factor));
        CHECK(std::abs(choice.speed - 1.0) <= pacing::kMaxSpeedChange);
        CHECK(choice.output_fps * choice.speed * choice.vsyncs_per_frame == Approx(hz).epsilon(1e-9));
      }
    }
  }
}

TEST_CASE("pacing : valeurs hors plage refusées", "[pacing]") {
  constexpr auto kOut = RefusalCode::FieldOutOfRange;
  CHECK(code_of(choose_cadence(fixed(120.0), 0.0, {2, 1}, 8.0)) == kOut);
  CHECK(code_of(choose_cadence(fixed(120.0), std::nan(""), {2, 1}, 8.0)) == kOut);
  CHECK(code_of(choose_cadence(fixed(0.0), kNtsc, {2, 1}, 8.0)) == kOut);
  CHECK(code_of(choose_cadence(fixed(2000.0), kNtsc, {2, 1}, 8.0)) == kOut);
  CHECK(code_of(choose_cadence(fixed(120.0), kNtsc, {1, 1}, 8.0)) == kOut);
  CHECK(code_of(choose_cadence(fixed(120.0), kNtsc, {9, 1}, 8.0)) == kOut);
  CHECK(code_of(choose_cadence(fixed(120.0), kNtsc, {4, 1}, 3.0)) == RefusalCode::FactorProductTooLarge);
  CHECK(code_of(fit_cadence(fixed(120.0), kNtsc, {0, 1})) == kOut);
  CHECK(code_of(fit_cadence(fixed(120.0), kNtsc, {17, 1})) == kOut);
  // VRR n'a pas besoin du rafraîchissement.
  CHECK(choose_cadence(vrr(0.0), kNtsc, {2, 1}, 8.0));
}

TEST_CASE("pacing : fit_cadence cale un facteur imposé, sinon les horodatages", "[pacing]") {
  auto plain = fit_cadence(fixed(120.0), kNtsc, {1, 1});
  REQUIRE(plain);
  CHECK(plain->grid == Grid::Vsync);
  CHECK(plain->vsyncs_per_frame == 5);
  CHECK(plain->speed == Approx(1.001).epsilon(1e-9));

  auto on_vrr = fit_cadence(vrr(120.0), kNtsc, {1, 1});
  REQUIRE(on_vrr);
  CHECK(on_vrr->grid == Grid::Timestamps);

  auto uneven = fit_cadence(fixed(60.0), 25.0, {1, 1});
  REQUIRE(uneven);
  CHECK(uneven->grid == Grid::Timestamps);
}

TEST_CASE("pacing : pace_rule ne retouche qu'une règle à une seule interpolation", "[pacing]") {
  SECTION("une interpolation : facteur remplacé") {
    const Rule rule{.steps = {interpolate({4, 1}), upscale(2)}};
    auto paced = pace_rule(rule, fixed(165.0), kNtsc);
    REQUIRE(paced);
    CHECK(paced->rule.steps[0].factor == Ratio{11011, 3200});
    CHECK(paced->rule.steps[1].factor == Ratio{2, 1});
    CHECK(paced->cadence.vsyncs_per_frame == 2);

    const Step down{.kind = StepKind::Downscale, .height = 720};
    auto reduced_first =
        pace_rule(Rule{.steps = {down, interpolate({4, 1}), upscale(2)}}, fixed(165.0), kNtsc);
    REQUIRE(reduced_first);
    CHECK(reduced_first->rule.steps[1].factor == Ratio{11011, 3200});
  }
  SECTION("deux interpolations : facteurs gardés, produit calé s'il le peut") {
    const Rule rule{.steps = {interpolate({3, 1}, Policy::Quality), interpolate({2, 1}), upscale(2)}};
    auto paced = pace_rule(rule, fixed(144.0), kNtsc);
    REQUIRE(paced);
    CHECK(paced->rule.steps[0].factor == Ratio{3, 1});
    CHECK(paced->rule.steps[1].factor == Ratio{2, 1});
    CHECK(paced->cadence.factor == Ratio{6, 1});
    CHECK(paced->cadence.grid == Grid::Vsync);
    CHECK(paced->cadence.vsyncs_per_frame == 1);

    auto judder = pace_rule(rule, fixed(165.0), kNtsc);
    REQUIRE(judder);
    CHECK(judder->rule.steps[0].factor == Ratio{3, 1});
    CHECK(judder->cadence.grid == Grid::Timestamps);
  }
  SECTION("aucune interpolation : la source seule, calée sur l'écran") {
    const Rule rule{.steps = {upscale(2)}};
    auto paced = pace_rule(rule, fixed(120.0), kNtsc);
    REQUIRE(paced);
    CHECK(paced->cadence.factor == Ratio{1, 1});
    CHECK(paced->cadence.vsyncs_per_frame == 5);
  }
  SECTION("le produit agrandissement × interpolation reste sous 16") {
    const Rule rule{.steps = {interpolate({4, 1}), upscale(4)}};
    auto paced = pace_rule(rule, fixed(120.0), kNtsc);
    REQUIRE(paced);
    CHECK(paced->rule.steps[0].factor == Ratio{5, 2});  // x5 ferait 20
  }
  SECTION("un terme de facteur hors bornes est refusé") {
    const Rule rule{.steps = {interpolate({4, 0}), interpolate({2, 1})}};
    auto paced = pace_rule(rule, fixed(120.0), kNtsc);
    REQUIRE_FALSE(paced);
    CHECK(paced.error().code == RefusalCode::FieldOutOfRange);
  }
}
