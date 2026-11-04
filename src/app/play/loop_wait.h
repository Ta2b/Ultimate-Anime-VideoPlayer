// L'attente de la boucle d'affichage jusqu'à l'image suivante (étape 6.7, essai --precise-wait) : une attente
// d'événement SDL jusqu'à 1,5 ms avant l'échéance (une image reçue, une touche, la réveillent), puis une
// attente absolue sur l'horloge monotone (std::this_thread::sleep_until, clock_nanosleep en TIMER_ABSTIME
// sous Linux), le relâchement des minuteurs du thread ramené à 1 ns. Sans l'essai : l'attente d'aujourd'hui,
// à la milliseconde, arrondie par défaut.
#pragma once

#include <atomic>
#include <cstdint>

namespace ajp::app::play {

// Le réveil de la boucle d'affichage par un message des autres threads : un événement SDL à elle, poussé une
// fois par attente (pending), qui fait rendre SDL_WaitEventTimeout.
struct LoopWake {
  std::atomic<bool> pending{false};
  std::uint32_t event = 0;  // 0 : SDL n'a plus d'événement à donner, l'attente va à son terme
  static void push(void* context);
};

// L'événement SDL du réveil, enregistré une fois.
std::uint32_t loop_wake_event();

// Attend au plus wait_us (au moins 1 ms sans l'essai), rend plus tôt sur un événement SDL.
void wait_loop(std::int64_t wait_us, bool precise);

}  // namespace ajp::app::play
