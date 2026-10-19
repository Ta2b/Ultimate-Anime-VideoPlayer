// Lecture de l'écran où se trouve la fenêtre : SDL3 partout, DXGI et DisplayConfig en plus
// sous Windows. À relire sur SDL_EVENT_WINDOW_DISPLAY_CHANGED, SDL_EVENT_DISPLAY_CURRENT_MODE_CHANGED
// et SDL_EVENT_WINDOW_HDR_STATE_CHANGED.
#pragma once

#include <expected>

#include "core/types.h"
#include "present/display_info.h"
#include "present/window.h"

namespace ajp::present {

// PresentFailed si SDL3 ne sait pas dire sur quel écran est la fenêtre ni son mode courant.
std::expected<SystemDisplay, Refusal> read_system_display(const Window& window);

}  // namespace ajp::present
