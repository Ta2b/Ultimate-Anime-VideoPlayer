// Le connecteur de l'écran (« DP-2 ») sous Wayland (C34-10) : deux écrans du même modèle portent le même nom
// pour SDL et le même nom d'EDID, seul le connecteur les sépare. SDL lit l'évènement name de wl_output mais
// ne le donne pas : on lie nos propres wl_output, sur notre file, et on retrouve celui de l'écran par sa
// position dans l'espace du compositeur.
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include <SDL3/SDL_video.h>

namespace ajp::app::play {

struct WaylandOutput {
  std::string name;  // « DP-2 » (wl_output v4) ; vide si le compositeur ne le dit pas
  int x = 0;         // wl_output.geometry : la position dans l'espace du compositeur
  int y = 0;
};

// La sortie à cette position, si une seule y est ; vide sinon. Pur.
std::string output_at(const std::vector<WaylandOutput>& outputs, int x, int y);

// Le nom de SDL suivi du connecteur entre parenthèses (« LG ULTRAGEAR (DP-2) ») sous Wayland quand il se lit,
// le nom de SDL seul sinon : pick_display trouve ainsi la sortie sans heuristique de noms. Pur.
std::string with_connector(std::string_view sdl_name, std::string_view connector);

// Le connecteur de l'écran display ; vide hors Wayland ou s'il ne se lit pas. Gardé par écran : le nom d'une
// sortie ne change pas, seule une position nouvelle la relit. Tout thread.
std::string wayland_connector(SDL_DisplayID display);

}  // namespace ajp::app::play
