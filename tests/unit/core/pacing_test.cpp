#include "core/pacing.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/phase_trim.h"

using Catch::Approx;
using namespace ajp::core;

namespace {

constexpr double kNtsc = 24000.0 / 1001.0;  // 23,976 im/s

}  // namespace

TEST_CASE("pacing : vitesse de calage sur l'écran, rien sans calage", "[pacing]") {
  CHECK(vsync_lock(60.0, kNtsc * 2.5).value_or(0.0) == Approx(1.001).epsilon(1e-12));   // ×5/2 : 1 vsync
  CHECK(vsync_lock(144.0, kNtsc * 2.0).value_or(0.0) == Approx(1.001).epsilon(1e-12));  // ×2 : 3 vsyncs
  CHECK(vsync_lock(59.94, kNtsc * 2.5).value_or(0.0) == Approx(1.0).epsilon(1e-5));
  CHECK_FALSE(vsync_lock(60.0, 24.0));          // 2,5 vsyncs : aucun entier
  CHECK_FALSE(vsync_lock(60.0, kNtsc * 3.0));   // plus rapide que l'écran
  CHECK_FALSE(vsync_lock(160.0, kNtsc * 2.0));  // 3,34 vsyncs : trop loin
  CHECK_FALSE(vsync_lock(0.0, kNtsc));          // écran inconnu
  CHECK_FALSE(vsync_lock(60.0, 0.0));           // sortie inconnue
  CHECK_FALSE(vsync_lock(1000.0, 23.976));      // au-delà de 16 vsyncs par image
}

TEST_CASE("pacing : petite fraction d'abord, sinon la fraction exacte, rien sous 1", "[pacing]") {
  CHECK(lock_factor(120.0 / kNtsc) == std::optional<Ratio>(Ratio{5, 1}));  // vitesse 1,001
  CHECK(lock_factor(60.0 / kNtsc) == std::optional<Ratio>(Ratio{5, 2}));   // 2,5025
  CHECK(lock_factor(165.0 / (2.0 * kNtsc)) == std::optional<Ratio>(Ratio{11011, 3200}));
  CHECK_FALSE(lock_factor(1.0));
  CHECK_FALSE(lock_factor(0.5));
}

namespace {

struct Slips {
  int count = 0;       // images doublées ou sautées
  double worst = 0.0;  // la plus grande position (fraction de période) après la première minute
};

// Une heure à 23,976 i/s ×6 sur un écran à 144 Hz dont la fréquence est mal connue de `error` (la vitesse
// calée sur la mesure laisse cette erreur relative) ; avec ou sans le calage de phase.
Slips one_hour(double error, bool trimmed) {
  const double period = 1e6 / 144.0;
  ajp::core::PhaseTrim trim;
  Slips out;
  double due = 0.2 * period;  // l'échéance de l'image, en µs d'hôte
  std::int64_t shown = 0;
  for (std::int64_t n = 0; due < 3'600e6; ++n) {
    const double x = due / period;
    const auto k = static_cast<std::int64_t>(std::llround(x));
    if (n > 0 && k - shown != 1) ++out.count;
    shown = k;
    if (trimmed) trim.note(x, static_cast<std::int64_t>(due));
    if (due > 60e6) out.worst = std::max(out.worst, std::abs(x - std::round(x)));
    // La vitesse réelle : celle du calage (une image par période) à l'erreur près, plus la correction.
    due += period / (1.0 + error + trim.trim());
  }
  return out;
}

}  // namespace

TEST_CASE("A12-06 : le calage de phase tient une heure sans image doublée ni sautée", "[pacing]") {
  // 0,002 Hz d'erreur à 144 Hz : 14 millionièmes, dans les deux sens.
  for (const double error : {+14e-6, -14e-6, +60e-6}) {
    INFO("erreur de fréquence " << error);
    const Slips before = one_hour(error, false);
    const Slips after = one_hour(error, true);
    CHECK(before.count >= 6);  // sans calage de phase : une image perdue toutes les huit minutes environ
    CHECK(after.count == 0);
    CHECK(after.worst < 0.35);  // l'échéance reste loin du milieu de deux vblanks
  }
  // Fréquence exacte : la correction reste nulle, la vitesse n'est jamais redemandée.
  ajp::core::PhaseTrim idle;
  int changes = 0;
  for (int n = 0; n < 10'000; ++n)
    if (idle.note(static_cast<double>(n) + 0.004, n * 6'944LL)) ++changes;
  CHECK(changes == 0);
  CHECK(idle.trim() == 0.0);
}

TEST_CASE("A12-06 : la correction est bornée, par pas, et au plus une fois par seconde", "[pacing]") {
  using ajp::core::PhaseTrim;
  PhaseTrim trim;
  int changes = 0;
  std::int64_t last = -10'000'000;
  for (int n = 0; n < 2'000; ++n) {
    const std::int64_t now = n * 6'944LL;
    if (trim.note(static_cast<double>(n) + 0.45, now)) {
      ++changes;
      CHECK(now - last >= PhaseTrim::kMinGapUs);
      last = now;
    }
    CHECK(std::abs(trim.trim()) <= PhaseTrim::kMaxTrim);
  }
  CHECK(changes >= 1);
  CHECK(trim.trim() == Catch::Approx(0.45 * PhaseTrim::kGain).margin(PhaseTrim::kStep));
  // Une image en retard sur son vblank : plus vite.
  CHECK(trim.trim() > 0.0);
  trim.reset();
  CHECK(trim.trim() == 0.0);
}
