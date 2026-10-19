// Période de vsync estimée sur des instants synthétiques : bruit d'ordonnanceur, images
// manquées, retards isolés, démarrage où la file se remplit d'un coup, changement de mode.
#include "present/vsync_estimator.h"

#include <cmath>
#include <cstdint>
#include <random>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "present/frame_vsyncs.h"

using ajp::present::VsyncEstimator;

namespace {

constexpr double kPeriod165 = 1e6 / 165.0;

struct Noise {
  std::mt19937 rng{20261019};
  std::normal_distribution<double> jitter{0.0, 80.0};  // 80 µs d'écart type
  std::uniform_real_distribution<double> unit{0.0, 1.0};
};

// Instant de la vsync n, vu par l'hôte : un retard fixe de 300 µs, plus le bruit.
std::int64_t observed(double period, std::int64_t n, Noise& noise) {
  return 5'000'000 +
         std::llround(static_cast<double>(n) * period + 300.0 + std::abs(noise.jitter(noise.rng)));
}

}  // namespace

TEST_CASE("rien avant 16 instants, puis la période à 0,01 % près malgré le bruit") {
  VsyncEstimator estimator;
  Noise noise;
  for (std::int64_t n = 0; n < 15; ++n) estimator.add(observed(kPeriod165, n, noise));
  CHECK_FALSE(estimator.estimate());
  for (std::int64_t n = 15; n < 400; ++n) estimator.add(observed(kPeriod165, n, noise));
  const auto estimate = estimator.estimate();
  REQUIRE(estimate);
  CHECK(estimate->period_us == Catch::Approx(kPeriod165).epsilon(1e-4));
  CHECK(estimate->refresh_hz() == Catch::Approx(165.0).epsilon(1e-4));
  CHECK(estimate->jitter_us < 150.0);
}

TEST_CASE("images manquées : un rang par vsync passée, la période ne change pas") {
  VsyncEstimator estimator;
  Noise noise;
  std::int64_t n = 0;
  for (int i = 0; i < 300; ++i) {
    // Une image sur cinq manque une ou deux vsyncs.
    n += noise.unit(noise.rng) < 0.2 ? (noise.unit(noise.rng) < 0.5 ? 2 : 3) : 1;
    estimator.add(observed(1e6 / 60.0, n, noise));
  }
  const auto estimate = estimator.estimate();
  REQUIRE(estimate);
  CHECK(estimate->period_us == Catch::Approx(1e6 / 60.0).epsilon(1e-4));
  // Les rangs suivent les vsyncs réelles, pas le nombre d'images.
  const auto index = estimator.index_at(observed(1e6 / 60.0, n + 10, noise));
  REQUIRE(index);
  CHECK(*index - estimate->anchor_index == 10);
}

TEST_CASE("retards isolés de l'ordonnanceur : écartés de la droite") {
  VsyncEstimator estimator;
  Noise noise;
  for (std::int64_t n = 0; n < 500; ++n) {
    std::int64_t host = observed(kPeriod165, n, noise);
    if (n % 37 == 0) host += 2500;  // préemption : 2,5 ms de retard
    estimator.add(host);
  }
  const auto estimate = estimator.estimate();
  REQUIRE(estimate);
  CHECK(estimate->period_us == Catch::Approx(kPeriod165).epsilon(1e-4));
  CHECK(estimate->samples < VsyncEstimator::kWindow);
}

TEST_CASE("démarrage : les premières images reviennent sans attendre") {
  VsyncEstimator estimator;
  Noise noise;
  // La swapchain a trois images libres : les trois premiers retours sont immédiats.
  for (std::int64_t i = 0; i < 3; ++i) estimator.add(1'000'000 + i * 40);
  for (std::int64_t n = 1; n < 200; ++n)
    estimator.add(1'000'000 + observed(1e6 / 120.0, n, noise) - 5'000'000);
  const auto estimate = estimator.estimate();
  REQUIRE(estimate);
  CHECK(estimate->period_us == Catch::Approx(1e6 / 120.0).epsilon(1e-4));
}

TEST_CASE("changement de mode : l'estimation repart sur la nouvelle grille") {
  VsyncEstimator estimator;
  Noise noise;
  for (std::int64_t n = 0; n < 200; ++n) estimator.add(observed(kPeriod165, n, noise));
  REQUIRE(estimator.estimate());
  const std::int64_t base = observed(kPeriod165, 200, noise);
  for (std::int64_t n = 0; n < 300; ++n) {
    estimator.add(base +
                  std::llround(static_cast<double>(n) * 1e6 / 60.0 + std::abs(noise.jitter(noise.rng))));
  }
  const auto estimate = estimator.estimate();
  REQUIRE(estimate);
  CHECK(estimate->period_us == Catch::Approx(1e6 / 60.0).epsilon(1e-4));
}

TEST_CASE("instants qui reculent ou se répètent : ignorés ; prévision des vsyncs") {
  VsyncEstimator estimator;
  for (std::int64_t n = 0; n < 64; ++n) {
    estimator.add(n * 10'000);
    estimator.add(n * 10'000);      // répété
    estimator.add(n * 10'000 - 1);  // recule
  }
  const auto estimate = estimator.estimate();
  REQUIRE(estimate);
  CHECK(estimate->period_us == Catch::Approx(10'000.0).epsilon(1e-9));
  CHECK(estimate->jitter_us == Catch::Approx(0.0).margin(1e-6));
  CHECK(estimator.index_at(1'004'000) == estimate->anchor_index + 37);
  CHECK(estimator.host_at(estimate->anchor_index + 2) == 650'000);
  estimator.reset();
  CHECK_FALSE(estimator.estimate());
  CHECK_FALSE(estimator.index_at(0));
}

TEST_CASE("vsync de chaque image FIFO : rangs croissants, numérotation continue") {
  using ajp::present::FrameVsyncs;
  VsyncEstimator estimator;
  FrameVsyncs frames;
  // Sans estimation : une vsync par image, l'instant d'acquisition.
  for (std::int64_t n = 0; n < 15; ++n) {
    estimator.add(n * 10'000);
    const auto tick = frames.next(estimator, n * 10'000, 3);
    CHECK(tick.index == n);
    CHECK(tick.host_us == n * 10'000);
    CHECK(tick.period_us == 0.0);
  }
  // Estimation : l'image paraît « latence » vsyncs plus tard, à l'instant prévu.
  estimator.add(150'000);
  auto tick = frames.next(estimator, 150'000, 3);
  CHECK(tick.index == 15);
  CHECK(tick.host_us == 180'000);
  CHECK(tick.period_us == Catch::Approx(10'000.0));
  estimator.add(160'000);
  tick = frames.next(estimator, 160'000, 3);
  CHECK(tick.index == 16);
  CHECK(tick.host_us == 190'000);
  // Images libres d'un coup (swapchain recréée) : une vsync chacune, dans l'ordre.
  for (int i = 1; i <= 3; ++i) CHECK(frames.next(estimator, 160'500, 3).index == 16 + i);
  // Estimation recommencée : ses rangs repartent de zéro, les nôtres continuent.
  estimator.reset();
  std::int64_t last = 19;
  for (std::int64_t n = 0; n < 40; ++n) {
    estimator.add(1'000'000 + n * 6'061);
    tick = frames.next(estimator, 1'000'000 + n * 6'061, 3);
    CHECK(tick.index == last + 1);
    last = tick.index;
  }
  CHECK(tick.period_us == Catch::Approx(6'061.0));
}
