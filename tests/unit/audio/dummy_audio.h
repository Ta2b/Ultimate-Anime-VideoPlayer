// Pilote audio « dummy » de SDL3 : il consomme la file en temps réel, sans
// matériel. Choisi par un indice posé dans le test, jamais par l'environnement.
#pragma once

#include <SDL3/SDL_hints.h>

namespace ajp::audio::test {

inline void use_dummy_audio() {
  SDL_SetHint(SDL_HINT_AUDIO_DRIVER, "dummy");
}

}  // namespace ajp::audio::test
