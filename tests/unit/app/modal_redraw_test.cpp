// La lecture pendant la boucle modale du système (modal_redraw.h, A10-05) : les expositions « en direct »
// que SDL envoie aux observateurs sont simulées par SDL_PushEvent, qui les appelle de la même façon.
#include "app/play/modal_redraw.h"

#include <chrono>
#include <cstdint>
#include <thread>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_init.h>
#include <catch2/catch_test_macros.hpp>

namespace ajp::app::play {
namespace {

// Une exposition de la fenêtre d'identifiant window ; live : celle de la boucle modale.
void expose(SDL_WindowID window, bool live) {
  SDL_Event event{};
  event.type = SDL_EVENT_WINDOW_EXPOSED;
  event.window.windowID = window;
  event.window.data1 = live ? 1 : 0;
  REQUIRE(SDL_PushEvent(&event));
}

void drain() {
  SDL_Event event;
  while (SDL_PollEvent(&event)) {
  }
}

}  // namespace

TEST_CASE("Boucle modale : seule une exposition en direct de notre fenêtre demande un tour", "[app][modal]") {
  const std::int64_t now = 1'000'000;
  CHECK(modal_turn_due(SDL_EVENT_WINDOW_EXPOSED, 1, true, now, 0));
  CHECK_FALSE(modal_turn_due(SDL_EVENT_WINDOW_EXPOSED, 0, true, now, 0));   // exposition ordinaire
  CHECK_FALSE(modal_turn_due(SDL_EVENT_WINDOW_EXPOSED, 1, false, now, 0));  // une autre fenêtre
  CHECK_FALSE(modal_turn_due(SDL_EVENT_WINDOW_RESIZED, 1, true, now, 0));
  CHECK_FALSE(modal_turn_due(SDL_EVENT_WINDOW_EXPOSED, 1, true, now, now - kModalTurnGapUs + 1));  // trop tôt
  CHECK(modal_turn_due(SDL_EVENT_WINDOW_EXPOSED, 1, true, now, now - kModalTurnGapUs));
}

TEST_CASE("Boucle modale : la lecture fait un tour à chaque exposition en direct, de l'intérieur de SDL",
          "[app][modal]") {
  REQUIRE(SDL_Init(SDL_INIT_EVENTS));
  int turns = 0;
  {
    // Sans fenêtre réelle : l'identifiant 0, que les événements du test portent aussi.
    ModalRedraw modal(nullptr, [&] {
      ++turns;
      expose(0, true);  // une exposition arrivée pendant le tour : pas de tour dans le tour
    });
    expose(0, false);  // la file ordinaire : la boucle du lecteur tourne déjà
    CHECK(turns == 0);
    expose(7, true);  // une autre fenêtre
    CHECK(turns == 0);
    expose(0, true);  // la boucle modale : SDL_PollEvent ne rend pas la main, le tour se fait d'ici
    CHECK(turns == 1);
    CHECK(modal.turns() == 1);
    expose(0, true);  // aussitôt après : au plus un tour par kModalTurnGapUs
    CHECK(turns == 1);
    std::this_thread::sleep_for(std::chrono::microseconds(2 * kModalTurnGapUs));
    expose(0, true);
    CHECK(turns == 2);
    // Un autre thread qui pousse le même événement ne fait pas tourner la lecture.
    std::this_thread::sleep_for(std::chrono::microseconds(2 * kModalTurnGapUs));
    std::thread([] { expose(0, true); }).join();
    CHECK(turns == 2);
    // Après une attente faite à l'intérieur d'un tour de lecture (tours suspendus), les tours reprennent.
    drain();
    expose(0, false);  // de quoi rendre l'attente aussitôt
    wait_events_no_modal_turn(1);
    expose(0, true);
    CHECK(turns == 3);  // l'attente finie, les tours reprennent
  }
  // Le garde détruit, plus rien n'est appelé.
  std::this_thread::sleep_for(std::chrono::microseconds(2 * kModalTurnGapUs));
  expose(0, true);
  CHECK(turns == 3);
  drain();
  SDL_Quit();
}

}  // namespace ajp::app::play
