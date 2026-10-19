// Les entrées d'essai de --keys : touches et souris poussées dans la file d'événements SDL, comme
// le clavier et la souris réels.
#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include <SDL3/SDL_video.h>

namespace ajp::app::play {

// « mouse=XxY » ou « click=XxY » : le point de la fenêtre, en fractions de sa taille.
std::optional<std::pair<float, float>> mouse_point(std::string_view text);
// Une touche (« space », « ctrl+2 »), ou la souris déplacée (« mouse=XxY ») ou cliquée
// (« click=XxY ») ; une touche inconnue est signalée au journal et ignorée.
void press(const std::string& name, SDL_Window* window);

}  // namespace ajp::app::play
