// MaxCLL et MaxFALL sur des images connues ; progression et temps restant à horloge pilotée.
#include <chrono>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "export/hdr10.h"
#include "export/progress.h"
#include "support.h"

namespace ajp::exporter {

using test::flat_frame;

TEST_CASE("courbe PQ : points de référence de ST 2084", "[export][hdr10]") {
  CHECK(pq_to_nits(0.0) == 0.0);
  CHECK(pq_to_nits(1.0) == Catch::Approx(10000.0).epsilon(1e-12));
  CHECK(pq_to_nits(0.5) == Catch::Approx(92.2457).epsilon(1e-5));
  CHECK(pq_to_nits(0.75183) == Catch::Approx(1000.0).epsilon(1e-3));
}

TEST_CASE("MaxCLL et MaxFALL justes sur des images connues", "[export][hdr10]") {
  LightMeter meter;
  CHECK(meter.level() == LightLevel{0, 0});

  meter.add(flat_frame(8, 4, 0, 0, 0).view());
  CHECK(meter.level() == LightLevel{0, 0});

  // Moitié des pixels à 10000 cd/m² (le rouge seul suffit), moitié au noir : MaxFALL 5000.
  auto half = flat_frame(8, 4, 0, 0, 0);
  for (std::size_t p = 0; p < 16; ++p) half.pixels[p * 4] = 65535;
  meter.add(half.view());
  CHECK(meter.level() == LightLevel{10000, 5000});

  // Une image uniforme plus faible ne change rien ; une image uniforme à 10000 monte MaxFALL.
  meter.add(flat_frame(8, 4, 32768, 0, 0).view());
  CHECK(meter.level() == LightLevel{10000, 5000});
  meter.add(flat_frame(8, 4, 0, 65535, 0).view());
  CHECK(meter.level() == LightLevel{10000, 10000});
}

TEST_CASE("la composante la plus forte fait la lumière du pixel", "[export][hdr10]") {
  LightMeter meter;
  // 32768 / 65535 en PQ : 92,25 cd/m², arrondi à 92 ; le canal alpha est ignoré.
  auto frame = flat_frame(4, 4, 100, 32768, 2000);
  meter.add(frame.view());
  CHECK(meter.level() == LightLevel{92, 92});

  // Une marge de ligne (stride) n'est pas lue.
  LightMeter padded;
  std::vector<std::uint16_t> pixels(2 * 6 * 4, 65535);  // 2 lignes de 6 pixels, dont 4 utiles
  for (std::size_t row = 0; row < 2; ++row) {
    for (std::size_t x = 0; x < 4; ++x) {
      for (std::size_t c = 0; c < 3; ++c) pixels[row * 24 + x * 4 + c] = 0;
    }
  }
  padded.add(HostFrame{.width = 4, .height = 2, .stride = 6 * 8, .rgba = pixels, .pts_us = 0});
  CHECK(padded.level() == LightLevel{0, 0});

  // Voie appareil : la mesure du backend entre telle quelle.
  meter.add(FrameLight{.max_nits = 400.4, .average_nits = 120.6});
  CHECK(meter.level() == LightLevel{400, 121});
}

TEST_CASE("progression : images, débit et temps restant", "[export][progress]") {
  using namespace std::chrono_literals;
  const ExportClock::time_point t0{};
  ProgressMeter meter(100);
  CHECK(meter.at(t0).frames_done == 0);
  CHECK_FALSE(meter.at(t0).remaining.has_value());
  meter.start(t0);

  ExportProgress p{};
  for (int i = 1; i <= 25; ++i) p = meter.frame_done(t0 + i * 100ms);  // 10 im/s
  CHECK(p.frames_done == 25);
  CHECK(p.frames_total == 100);
  CHECK(p.frames_per_second == Catch::Approx(10.0));
  CHECK(p.elapsed == 2500ms);
  REQUIRE(p.remaining.has_value());
  CHECK(*p.remaining == 7500ms);
  CHECK_FALSE(p.finishing);

  // Débit moyen depuis le départ : un arrêt de 2,5 s le divise par deux.
  p = meter.at(t0 + 5000ms);
  CHECK(p.frames_per_second == Catch::Approx(5.0));
  CHECK(*p.remaining == 15000ms);
}

TEST_CASE("progression sans total connu : pas de temps restant", "[export][progress]") {
  using namespace std::chrono_literals;
  const ExportClock::time_point t0{};
  ProgressMeter meter(std::nullopt);
  meter.start(t0);
  const auto p = meter.frame_done(t0 + 50ms);
  CHECK(p.frames_done == 1);
  CHECK_FALSE(p.frames_total.has_value());
  CHECK_FALSE(p.remaining.has_value());
  CHECK(p.frames_per_second == Catch::Approx(20.0));

  // Plus d'images que le total annoncé : le temps restant reste nul, jamais négatif.
  ProgressMeter over(1);
  over.start(t0);
  over.frame_done(t0 + 10ms);
  CHECK(*over.frame_done(t0 + 20ms).remaining == 0ms);
}

}  // namespace ajp::exporter
