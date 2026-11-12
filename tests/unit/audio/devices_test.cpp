// La désignation du défaut du système dans la liste des appareils (C16-09).
#include "audio/devices.h"

#include <vector>

#include <catch2/catch_test_macros.hpp>

using ajp::audio::AudioDevice;
using ajp::audio::mark_default_device;
using ajp::audio::resolve_device;

namespace {

std::vector<AudioDevice> devices() {
  return {{.id = 1, .name = "HDMI", .channels = 8, .sample_rate = 192'000},
          {.id = 2, .name = "USB DAC", .channels = 2, .sample_rate = 48'000},
          {.id = 3, .name = "USB DAC", .channels = 2, .sample_rate = 96'000}};
}

}  // namespace

TEST_CASE("mark_default_device : un seul appareil, le premier de ce nom", "[audio][devices]") {
  auto list = devices();
  mark_default_device(list, "USB DAC");
  CHECK_FALSE(list[0].is_default);
  CHECK(list[1].is_default);
  CHECK_FALSE(list[2].is_default);  // le jumeau n'est pas aussi le défaut
}

TEST_CASE("mark_default_device : sans appareil de ce nom, personne n'est le défaut", "[audio][devices]") {
  auto list = devices();
  mark_default_device(list, "Casque");
  for (const AudioDevice& device : list) CHECK_FALSE(device.is_default);
  mark_default_device(list, "");
  for (const AudioDevice& device : list) CHECK_FALSE(device.is_default);
  // Le HDMI 7.1 en tête de liste n'est pas pris pour le défaut.
  CHECK_FALSE(resolve_device(list, ""));
}
