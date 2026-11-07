#include "core/pacing.h"

#include <optional>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using namespace ajp::core;

namespace {

constexpr double kNtsc = 24000.0 / 1001.0;  // 23,976 im/s

}  // namespace

TEST_CASE("pacing : vitesse de calage sur l'écran, 1 sans calage", "[pacing]") {
  CHECK(vsync_speed(60.0, kNtsc * 2.5) == Approx(1.001).epsilon(1e-12));   // ×5/2 : 1 vsync
  CHECK(vsync_speed(144.0, kNtsc * 2.0) == Approx(1.001).epsilon(1e-12));  // ×2 : 3 vsyncs
  CHECK(vsync_speed(59.94, kNtsc * 2.5) == Approx(1.0).epsilon(1e-5));
  CHECK(vsync_speed(60.0, 24.0) == 1.0);          // 2,5 vsyncs : aucun entier
  CHECK(vsync_speed(60.0, kNtsc * 3.0) == 1.0);   // plus rapide que l'écran
  CHECK(vsync_speed(160.0, kNtsc * 2.0) == 1.0);  // 3,34 vsyncs : trop loin
  CHECK(vsync_speed(0.0, kNtsc) == 1.0);          // écran inconnu
  CHECK(vsync_speed(60.0, 0.0) == 1.0);           // sortie inconnue
  CHECK(vsync_speed(1000.0, 23.976) == 1.0);      // au-delà de 16 vsyncs par image
}

TEST_CASE("pacing : petite fraction d'abord, sinon la fraction exacte, rien sous 1", "[pacing]") {
  CHECK(lock_factor(120.0 / kNtsc) == std::optional<Ratio>(Ratio{5, 1}));  // vitesse 1,001
  CHECK(lock_factor(60.0 / kNtsc) == std::optional<Ratio>(Ratio{5, 2}));   // 2,5025
  CHECK(lock_factor(165.0 / (2.0 * kNtsc)) == std::optional<Ratio>(Ratio{11011, 3200}));
  CHECK_FALSE(lock_factor(1.0));
  CHECK_FALSE(lock_factor(0.5));
}
