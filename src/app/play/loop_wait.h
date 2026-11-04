// L'attente de la boucle d'affichage jusqu'à l'image suivante (étape 6.7, essai --precise-wait) : une attente
// d'événement SDL jusqu'à 1,5 ms avant l'échéance (une image reçue, une touche, la réveillent), puis une
// attente absolue sur l'horloge monotone (std::this_thread::sleep_until, clock_nanosleep en TIMER_ABSTIME
// sous Linux), le relâchement des minuteurs du thread ramené à 1 ns. Sans l'essai : l'attente d'aujourd'hui,
// à la milliseconde, arrondie par défaut.
#pragma once

#include <cstdint>

namespace ajp::app::play {

// Attend au plus wait_us (au moins 1 ms sans l'essai), rend plus tôt sur un événement SDL.
void wait_loop(std::int64_t wait_us, bool precise);

}  // namespace ajp::app::play
