#include "audio/clock.h"

#include <atomic>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/clock.h"

using ajp::ClockSnapshot;
using ajp::audio::AudioClock;

TEST_CASE("ClockSnapshot : extrapole le temps média selon sa vitesse", "[audio][clock]") {
  const ClockSnapshot playing{.media_us = 10'000'000, .host_us = 500'000, .rate = 1.0, .valid = true};
  CHECK(playing.media_at(500'000) == 10'000'000);
  CHECK(playing.media_at(1'500'000) == 11'000'000);
  CHECK(playing.media_at(0) == 9'500'000);

  const ClockSnapshot paused{.media_us = 10'000'000, .host_us = 500'000, .rate = 0.0, .valid = true};
  CHECK(paused.media_at(60'000'000) == 10'000'000);

  const ClockSnapshot ntsc{.media_us = 0, .host_us = 0, .rate = 1.001, .valid = true};
  CHECK(ntsc.media_at(1'000'000) == 1'001'000);
}

TEST_CASE("ajp::host_now_us : horloge monotone", "[audio][clock]") {
  const std::int64_t first = ajp::host_now_us();
  const std::int64_t second = ajp::host_now_us();
  CHECK(second >= first);
}

TEST_CASE("AudioClock : un écrivain, plusieurs lecteurs, aucune valeur déchirée", "[audio][clock]") {
  AudioClock clock;
  CHECK_FALSE(clock.snapshot().valid);
  std::atomic<int> torn{0};  // Catch2 n'accepte pas d'assertion hors du thread du test
  std::jthread writer([&clock] {
    for (std::int64_t i = 1; i <= 10'000; ++i) {
      clock.publish({.media_us = i, .host_us = 2 * i, .rate = 1.0, .valid = true});
    }
  });
  std::vector<std::jthread> readers;
  for (int r = 0; r < 3; ++r) {
    readers.emplace_back([&clock, &torn] {
      for (int i = 0; i < 10'000; ++i) {
        const ClockSnapshot snapshot = clock.snapshot();
        if (snapshot.valid && snapshot.host_us != 2 * snapshot.media_us) ++torn;
      }
    });
  }
  writer.join();
  readers.clear();
  CHECK(torn == 0);
  CHECK(clock.snapshot().media_us == 10'000);
}
