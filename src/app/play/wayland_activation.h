// Fenêtre passée devant sous Wayland (C34-15) : KWin et Mutter ne donnent le focus qu'à une fenêtre qui
// présente un jeton d'activation. Le lancement qui transmet ses fichiers au lecteur ouvert reçoit ce jeton
// de son lanceur (XDG_ACTIVATION_TOKEN, gestionnaire de fichiers, menu) et le lui passe ; le lecteur
// l'utilise pour sa propre surface par xdg_activation_v1.
#pragma once

#include <string_view>

struct SDL_Window;

namespace ajp::app::play {

// Active la fenêtre avec le jeton. Faux hors Wayland, sans le protocole chez le compositeur, ou sans jeton :
// l'appelant garde alors son repli (SDL_RaiseWindow et le signalement dans la barre des tâches). Thread de la
// fenêtre seulement.
bool wayland_activate(SDL_Window* window, std::string_view token);

}  // namespace ajp::app::play
