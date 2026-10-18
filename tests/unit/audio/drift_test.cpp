// Le régulateur de dérive, en simulation : un périphérique qui lit sa file au
// rapport demandé, une horloge maître qui avance à 1 + offset, des mesures
// bruitées toutes les 1 024 trames à 48 kHz.
#include "audio/drift.h"

#include <cmath>
#include <cstdint>

#include <catch2/catch_test_macros.hpp>

#include "audio/pcm.h"

using ajp::audio::DriftRegulator;
using ajp::audio::limits::kMaxDriftCorrection;

namespace {

constexpr double kPeriodS = 1024.0 / 48'000.0;
constexpr std::int64_t kPeriodUs = 21'333;

// Bruit pseudo-aléatoire reproductible, uniforme dans [-1, 1].
class Noise {
 public:
  double next() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(state_ >> 11) / static_cast<double>(1ULL << 52) - 1.0;
  }

 private:
  std::uint64_t state_ = 42;
};

struct Run {
  double final_error_s = 0.0;
  double final_ratio = 1.0;
  double worst_correction = 0.0;
};

// offset : avance relative de l'horloge maître (0,001 pour 1,001) ; error_s : écart de départ.
Run simulate(DriftRegulator& regulator, double offset, double error_s, double seconds, double noise_s) {
  Noise noise;
  Run run;
  for (double t = 0.0; t < seconds; t += kPeriodS) {
    const double measured = error_s + noise_s * noise.next();
    const double ratio = regulator.update(std::llround(measured * 1e6), kPeriodUs);
    run.worst_correction = std::max(run.worst_correction, std::abs(ratio - 1.0));
    error_s += (ratio - (1.0 + offset)) * kPeriodS;  // l'audio avance au rapport, le maître à 1 + offset
  }
  run.final_error_s = error_s;
  run.final_ratio = regulator.ratio();
  return run;
}

constexpr double kBound = kMaxDriftCorrection + 1e-12;

}  // namespace

TEST_CASE("DriftRegulator : sans écart ni dérive, aucune correction", "[audio][drift]") {
  DriftRegulator regulator;
  for (int i = 0; i < 100; ++i) CHECK(regulator.update(0, kPeriodUs) == 1.0);
}

TEST_CASE("DriftRegulator : la dérive NTSC 1,001 est apprise, l'écart converge", "[audio][drift]") {
  DriftRegulator regulator;
  const Run run = simulate(regulator, 0.001, 0.030, 180.0, 0.002);
  CHECK(run.worst_correction <= kBound);
  CHECK(std::abs(run.final_error_s) < 0.001);
  CHECK(std::abs(run.final_ratio - 1.001) < 0.0001);
}

TEST_CASE("DriftRegulator : un maître plus lent ralentit l'audio", "[audio][drift]") {
  DriftRegulator regulator;
  const Run run = simulate(regulator, -0.001, -0.030, 180.0, 0.002);
  CHECK(run.worst_correction <= kBound);
  CHECK(std::abs(run.final_error_s) < 0.001);
  CHECK(std::abs(run.final_ratio - 0.999) < 0.0001);
}

TEST_CASE("DriftRegulator : au-delà de 0,125 %, la correction sature sans dépasser la borne",
          "[audio][drift]") {
  DriftRegulator regulator;
  const Run run = simulate(regulator, 0.003, 0.0, 60.0, 0.0);
  CHECK(run.worst_correction <= kBound);
  CHECK(std::abs(run.final_ratio - (1.0 + kMaxDriftCorrection)) < 1e-12);
  // L'écart grandit : le lecteur recalera franchement au-delà de son seuil.
  CHECK(run.final_error_s < -0.05);
}

TEST_CASE("DriftRegulator : un écart énorme reste borné dès la première mesure", "[audio][drift]") {
  DriftRegulator ahead;
  CHECK(std::abs(ahead.update(10'000'000, 0) - (1.0 - kMaxDriftCorrection)) < 1e-12);
  DriftRegulator behind;
  CHECK(std::abs(behind.update(-10'000'000, 0) - (1.0 + kMaxDriftCorrection)) < 1e-12);
}

TEST_CASE("DriftRegulator : restart garde la dérive apprise, reset l'oublie", "[audio][drift]") {
  DriftRegulator regulator;
  simulate(regulator, 0.001, 0.0, 180.0, 0.0);
  regulator.restart();
  CHECK(std::abs(regulator.update(0, kPeriodUs) - 1.001) < 0.0001);
  regulator.reset();
  CHECK(regulator.update(0, kPeriodUs) == 1.0);
}
