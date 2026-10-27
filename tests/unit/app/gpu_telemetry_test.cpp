// La télémétrie NVML : l'UUID tel que NVML le nomme, les raisons d'écrêtage, la ligne de l'OSD et
// l'avertissement de sous-cadencement. NVML lui-même n'est pas appelé ici.
#include "app/play/gpu_telemetry.h"

#include <catch2/catch_test_macros.hpp>

namespace ajp::app::play {
namespace {

TEST_CASE("UUID NVML : GPU- et cinq groupes d'octets", "[app][telemetry]") {
  DeviceUuid uuid{};
  for (std::size_t i = 0; i < uuid.size(); ++i) uuid[i] = static_cast<std::uint8_t>(0x10 + i);
  CHECK(nvml_uuid_text(uuid) == "GPU-10111213-1415-1617-1819-1a1b1c1d1e1f");
}

TEST_CASE("Raisons d'écrêtage : puissance et chaleur comptent, le ralenti non", "[app][telemetry]") {
  CHECK(throttle_text(0x1 | 0x2).empty());  // GpuIdle, ApplicationsClocksSetting
  CHECK(throttle_text(0x4) == "limite de puissance");
  CHECK(throttle_text(0x4 | 0x20) == "limite de puissance + chaleur");
  CHECK(throttle_text(0x40 | 0x80) == "chaleur + frein de puissance");
}

TEST_CASE("Version du pilote : plus ancien que, et prudence si illisible", "[app][telemetry]") {
  CHECK(nvidia_driver_older_than("535.104.05", 550));
  CHECK_FALSE(nvidia_driver_older_than("615.78.08", 550));
  CHECK(nvidia_driver_older_than("580.65.06", 580, 82, 7));
  CHECK_FALSE(nvidia_driver_older_than("580.82.07", 580, 82, 7));
  CHECK_FALSE(nvidia_driver_older_than("580.82.09", 580, 82, 7));
  CHECK_FALSE(nvidia_driver_older_than("Mesa 25.1.4", 550));
  CHECK_FALSE(nvidia_driver_older_than("", 550));
}

TEST_CASE("Ligne de l'OSD et avertissement de sous-cadencement", "[app][telemetry]") {
  GpuSample s{.sm_mhz = 2520,
              .sm_max_mhz = 3105,
              .mem_mhz = 10501,
              .pstate = 2,
              .power_mw = 188000,
              .power_limit_mw = 285000,
              .temperature_c = 61,
              .gpu_percent = 93,
              .pcie_generation = 4,
              .pcie_width = 16};
  CHECK(telemetry_line(s) == "GPU 2520 / 3105 MHz · P2 · 188 / 285 W · 61 °C · 93 % · PCIe 4 x16");
  CHECK(telemetry_warning(s).empty());
  s.sm_mhz = 1200;
  CHECK(telemetry_warning(s).find("horloges basses") != std::string::npos);
  s.gpu_percent = 30;  // peu chargée : descendre en horloge est normal
  CHECK(telemetry_warning(s).empty());
  s.throttle = 0x4;
  CHECK(telemetry_line(s).ends_with(" · limite de puissance"));
}

}  // namespace
}  // namespace ajp::app::play
