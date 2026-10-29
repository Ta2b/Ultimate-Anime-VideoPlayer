#include "app/play/scripted_input.h"

#include <charconv>
#include <deque>
#include <string>
#include <system_error>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_keyboard.h>

#include "core/charconv.h"
#include "core/log.h"
#include "prepare/utf8.h"

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
  if (core::from_chars(xs.data(), xs.data() + xs.size(), x).ec != std::errc{} ||
      core::from_chars(ys.data(), ys.data() + ys.size(), y).ec != std::errc{})
    return std::nullopt;
  return std::pair{x, y};
}

// Touche pressée d'office (--keys) : un événement SDL de plus dans la file, comme le clavier.
// « mouse=XxY » déplace la souris (la barre de lecture paraît), « click=XxY » y clique, « wheel=N » tourne la
// molette de N crans (négatif : vers le bas) là où est la souris (les pages du menu se parcourent ainsi).
void press(const std::string& name, SDL_Window* window) {
  if (name.starts_with("drop=")) {           // un fichier glissé sur la fenêtre
    static std::deque<std::string> dropped;  // SDL garde le pointeur jusqu'à la lecture de l'événement
    dropped.push_back(name.substr(5));
    SDL_Event drop{};
    drop.type = SDL_EVENT_DROP_FILE;
    drop.drop.windowID = SDL_GetWindowID(window);
    drop.drop.data = dropped.back().c_str();
    log_channel("app")->info("--keys : fichier glissé {}", dropped.back());
    SDL_PushEvent(&drop);
    return;
  }
  if (name.starts_with("wheel=")) {
    int notches = 0;
    const std::string_view text = std::string_view(name).substr(6);
    if (core::from_chars(text.data(), text.data() + text.size(), notches).ec != std::errc{}) return;
    float x = 0.0F;
    float y = 0.0F;
    SDL_GetMouseState(&x, &y);
    SDL_Event wheel{};
    wheel.type = SDL_EVENT_MOUSE_WHEEL;
    wheel.wheel.windowID = SDL_GetWindowID(window);
    wheel.wheel.y = static_cast<float>(notches);
    wheel.wheel.mouse_x = x;
    wheel.wheel.mouse_y = y;
    log_channel("app")->info("--keys : molette {}", notches);
    SDL_PushEvent(&wheel);
    return;
  }
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

std::optional<ExportJob> scripted_export(std::string_view name, const PlaySettings& running,
                                         const media::TrackInfo& video, const std::filesystem::path& source) {
  constexpr std::string_view kExport = "export";
  if (name != kExport && !name.starts_with("export=")) return std::nullopt;
  ExportJob job{.chain = running};
  const ExportSize size = export_size(chain_output(job.chain, video.width, video.height),
                                      {.width = video.width, .height = video.height}, job.height);
  job.output = name.size() > kExport.size() + 1
                   ? prepare::path_from_utf8(name.substr(kExport.size() + 1))
                   : default_export_path(source, size, export_rate(video.frame_rate, factor_of(job.chain)));
  log_channel("app")->info("--keys : export vers {}", prepare::utf8_path(job.output));
  return job;
}

}  // namespace ajp::app::play
