// Lecture de l'écran par DXGI et DisplayConfig, Windows seulement (comme mpv : HDR actif et
// luminances par DXGI_OUTPUT_DESC1, blanc SDR par DisplayConfigGetDeviceInfo).
// En-tête interne, inclus par display_sdl.cpp sous _WIN32 seulement.
#pragma once

#include <SDL3/SDL_video.h>

#include "present/display_info.h"

namespace ajp::present {

// Complète out pour l'écran de la fenêtre ; ce qui ne se lit pas entre dans out.unknown.
void read_dxgi(SDL_Window* window, SystemDisplay& out);

}  // namespace ajp::present
