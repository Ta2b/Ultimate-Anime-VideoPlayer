#include "app/play/scripted_input.h"

#include <charconv>
#include <system_error>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_keyboard.h>

#include "core/log.h"

namespace ajp::app::play {

// « mouse=XxY » ou « click=XxY » de --keys : la souris passe (et clique) à ce point de la
// fenêtre (fractions de sa taille).
std::optional<std::pair<float, float>> mouse_point(std::string_view text) {
  if (!text.starts_with("mouse=") && !text.starts_with("click=")) return std::nullopt;
  text.remove_prefix(6);
  const std::size_t cross = text.find('x');
  if (cross == std::string_view::npos) return std::nullopt;
  float x = 0.0F;
  float y = 0.0F;
  const std::string_view xs = text.substr(0, cross);
  const std::string_view ys = text.substr(cross + 1);
  if (std::from_chars(xs.data(), xs.data() + xs.size(), x).ec != std::errc{} ||
      std::from_chars(ys.data(), ys.data() + ys.size(), y).ec != std::errc{})
    return std::nullopt;
  return std::pair{x, y};
}

// Touche pressée d'office (--keys) : un événement SDL de plus dans la file, comme le clavier.
// « mouse=XxY » déplace la souris (la barre de lecture paraît), « click=XxY » y clique.
void press(const std::string& name, SDL_Window* window) {
  if (const auto point = mouse_point(name)) {
    int w = 0;
    int h = 0;
    SDL_GetWindowSize(window, &w, &h);
    SDL_Event motion{};
    motion.type = SDL_EVENT_MOUSE_MOTION;
    motion.motion.windowID = SDL_GetWindowID(window);
    motion.motion.x = point->first * static_cast<float>(w);
    motion.motion.y = point->second * static_cast<float>(h);
    log_channel("app")->info("--keys : souris en {:.0f}, {:.0f}", motion.motion.x, motion.motion.y);
    SDL_PushEvent(&motion);
    if (name.starts_with("click=")) {
      for (const bool down : {true, false}) {
        SDL_Event button{};
        button.type = down ? SDL_EVENT_MOUSE_BUTTON_DOWN : SDL_EVENT_MOUSE_BUTTON_UP;
        button.button.windowID = motion.motion.windowID;
        button.button.button = SDL_BUTTON_LEFT;
        button.button.down = down;
        button.button.clicks = 1;
        button.button.x = motion.motion.x;
        button.button.y = motion.motion.y;
        SDL_PushEvent(&button);
      }
    }
    return;
  }
  SDL_Event event{};
  event.type = SDL_EVENT_KEY_DOWN;
  event.key.windowID = SDL_GetWindowID(window);
  std::string_view key = name;
  if (key.starts_with("ctrl+")) {  // « ctrl+2 » : Ctrl tenu
    event.key.mod = SDL_KMOD_LCTRL;
    key.remove_prefix(5);
  }
  event.key.key = SDL_GetKeyFromName(std::string(key).c_str());
  event.key.down = true;
  if (event.key.key == SDLK_UNKNOWN) {
    log_channel("app")->warn("--keys : touche inconnue « {} »", name);
    return;
  }
  log_channel("app")->info("--keys : {}", name);
  SDL_PushEvent(&event);
}

}  // namespace ajp::app::play
