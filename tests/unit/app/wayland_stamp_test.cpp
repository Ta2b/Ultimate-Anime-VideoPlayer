// Les horodatages du compositeur (placebo_wayland.h, chantier 0.7) : secondes sur 64 bits, MSC, drapeaux.
#include <cstdint>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "app/play/placebo_wayland.h"

using namespace ajp::app::play;

TEST_CASE("wp_presentation : un horodatage en clair", "[app][wayland]") {
  // 2^32 + 5 secondes et 123 456 789 ns, MSC 2^32 + 7, vsync | horloge matérielle (0x1 | 0x2).
  const ScreenStamp s = stamp_of(42, 1, 5, 123'456'789, 16'666'667, 1, 7, 0x3);
  CHECK(s.seq == 42);
  CHECK(s.presented_us == ((std::int64_t{1} << 32) + 5) * 1'000'000 + 123'456);
  CHECK(s.refresh_ns == 16'666'667);
  CHECK(s.msc == (std::uint64_t{1} << 32) + 7);
  CHECK(s.flags == 0x3);
  // Une heure ordinaire : la moitié haute à 0.
  CHECK(stamp_of(1, 0, 1000, 999'999, 0, 0, 3, 0).presented_us == 1'000'000'999);
  CHECK(std::string(screen_stamp_header()) == "seq\tpresented_us\trefresh_ns\tmsc\tflags\n");
  CHECK(screen_stamp_line(stamp_of(9, 0, 2, 500'000, 6'944'444, 0, 120, 0x7)) ==
        "9\t2000500\t6944444\t120\t0x7\n");
}
