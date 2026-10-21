// Retard des images de sortie : médiane des dernières mesures, et l'instant média où une image
// soumise maintenant sera prête.
#include "app/play/frame_latency.h"

#include <catch2/catch_test_macros.hpp>

namespace ajp::app::play {
namespace {

TEST_CASE("Retard des images : médiane des dernières mesures, un à-coup isolé ne compte pas",
          "[app][latency]") {
  FrameLatency latency;
  CHECK(latency.us() == 0);  // aucune mesure
  latency.observe(20'000);
  CHECK(latency.us() == 20'000);
  for (int i = 0; i < 4; ++i) latency.observe(20'000);
  latency.observe(900'000);  // premier passage d'un moteur
  CHECK(latency.us() == 20'000);
  // Hors bornes : ignorées.
  latency.observe(-1);
  latency.observe(FrameLatency::kMaxSampleUs + 1);
  CHECK(latency.us() == 20'000);
  // Une surcharge qui dure : la médiane la suit en quelques images.
  for (int i = 0; i < 3; ++i) latency.observe(120'000);
  CHECK(latency.us() == 20'000);  // cinq à 20 ms, une à 900 ms, trois à 120 ms
  latency.observe(120'000);       // la plus ancienne sort : quatre à 20 ms, quatre à 120 ms
  CHECK(latency.us() == 120'000);
  for (std::size_t i = 0; i < FrameLatency::kSamples; ++i) latency.observe(30'000);
  CHECK(latency.us() == 30'000);  // les anciennes sont sorties de la fenêtre
}

TEST_CASE("Retard des images : prête une période plus loin, ou après le retard mesuré", "[app][latency]") {
  const ClockSnapshot now{.media_us = 1'000'000, .host_us = 500'000, .rate = 1.0, .epoch = 0, .valid = true};
  // Sans retard mesuré, ou plus court qu'une période : une période plus loin.
  CHECK(ready_media_us(now, 510'000, 41'667, 0) == 1'051'667);
  CHECK(ready_media_us(now, 510'000, 41'667, 30'000) == 1'051'667);
  // GPU débordé : le retard l'emporte.
  CHECK(ready_media_us(now, 510'000, 41'667, 150'000) == 1'160'000);
  // À vitesse double, le retard compte double en temps média.
  const ClockSnapshot fast{.media_us = 1'000'000, .host_us = 500'000, .rate = 2.0, .epoch = 0, .valid = true};
  CHECK(ready_media_us(fast, 500'000, 0, 100'000) == 1'200'000);
}

}  // namespace
}  // namespace ajp::app::play
