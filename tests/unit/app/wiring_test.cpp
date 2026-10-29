// Horloges du câblage : l'horloge hôte à la vitesse qui cale l'écran, sa reprise de la position
// de l'audio, et l'horloge que lisent la session et le thread gpu (l'audio ou l'hôte).
#include "app/play/wiring.h"

#include <memory>

#include <catch2/catch_test_macros.hpp>

#include "doubles.h"

namespace ajp::app::play {
namespace {

TEST_CASE("Qui mène : l'audio, sauf vitesse de calage, audio perdu ou son en avance de plus de 60 ms",
          "[app][clock]") {
  CHECK_FALSE(host_must_lead(1.0, false, 0));
  CHECK_FALSE(host_must_lead(1.0, false, 150'000));  // son en retard : l'audio mène
  CHECK_FALSE(host_must_lead(1.0, false, -59'000));
  CHECK(host_must_lead(1.0, false,
                       -60'000));  // le son avant l'image demande plus d'images que la file n'en tient
  CHECK(host_must_lead(1.0, false, -300'000));
  CHECK(host_must_lead(1.0, false, -5'000'000));
  CHECK(host_must_lead(0.999, false, 0));
  CHECK(host_must_lead(1.0, true, 0));
}

TEST_CASE("Horloge hôte : la vitesse vaut tout de suite en marche, au départ sinon", "[app][clock]") {
  HostClock clock;
  clock.set_rate(1.001, 0);  // pas encore partie : gardée pour le départ
  CHECK_FALSE(clock.snapshot().valid);
  clock.start(1, 1'000'000, 0);
  CHECK(clock.snapshot().rate == 1.001);
  CHECK(clock.snapshot().media_at(1'000'000) == 2'001'000);
  clock.set_rate(1.0, 1'000'000);  // repart de sa position à cet instant, sans saut
  CHECK(clock.snapshot().media_at(1'000'000) == 2'001'000);
  CHECK(clock.snapshot().media_at(2'000'000) == 3'001'000);
  clock.pause(2'000'000);
  clock.set_rate(1.001, 2'500'000);  // en pause : rien ne bouge
  CHECK(clock.snapshot().rate == 0.0);
  CHECK(clock.snapshot().media_at(9'000'000) == 3'001'000);
  clock.start(1, 3'001'000, 3'000'000);
  CHECK(clock.snapshot().rate == 1.001);
  clock.pause(4'000'000);                      // à 4'002'000
  clock.start(1, 3'990'000, 5'000'000, true);  // même époque : reprend d'où elle s'est arrêtée
  CHECK(clock.snapshot().media_at(5'000'000) == 4'002'000);
  clock.pause(5'000'000);
  clock.start(2, 7'000'000, 6'000'000, true);  // autre époque : part du pts donné
  CHECK(clock.snapshot().media_at(6'000'000) == 7'000'000);
  clock.flush(2);
  CHECK_FALSE(clock.snapshot().valid);
  CHECK(clock.snapshot().epoch == 2);
}

TEST_CASE("Horloge hôte : reprend la position de l'audio, en marche ou en pause", "[app][clock]") {
  HostClock clock;
  clock.set_rate(1.001, 0);
  const ClockSnapshot audio{.media_us = 500'000, .host_us = 100'000, .rate = 1.0, .epoch = 3, .valid = true};
  clock.adopt(audio, true, 200'000);
  CHECK(clock.snapshot().epoch == 3);
  CHECK(clock.snapshot().media_at(200'000) == 600'000);
  CHECK(clock.snapshot().rate == 1.001);
  clock.adopt(audio, false, 200'000);  // en pause : arrêtée à la même position
  CHECK(clock.snapshot().rate == 0.0);
  CHECK(clock.snapshot().media_at(900'000) == 600'000);
  clock.adopt(ClockSnapshot{.epoch = 4}, true, 300'000);  // audio pas encore parti : position inconnue
  CHECK_FALSE(clock.snapshot().valid);
  CHECK(clock.snapshot().epoch == 4);
}

TEST_CASE("Horloge menante : l'audio, puis l'hôte quand l'écran mène", "[app][clock]") {
  auto audio = std::make_shared<test::FakeClock>();
  auto host = std::make_shared<HostClock>();
  audio->value = {.media_us = 10, .valid = true};
  host->start(0, 20, 0);
  LeadClock lead(audio, host);
  CHECK_FALSE(lead.host_leads());
  CHECK(lead.snapshot().media_us == 10);
  lead.lead_with_host(true);
  CHECK(lead.snapshot().media_us == 20);
  lead.lead_with_host(false);
  CHECK(lead.snapshot().media_us == 10);
}

}  // namespace
}  // namespace ajp::app::play
