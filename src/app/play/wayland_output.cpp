#include "app/play/wayland_output.h"

#include <format>

#ifdef AJP_WITH_WAYLAND_PRESENTATION
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <wayland-client.h>

#include <SDL3/SDL_properties.h>
#endif

namespace ajp::app::play {

std::string output_at(const std::vector<WaylandOutput>& outputs, int x, int y) {
  const WaylandOutput* found = nullptr;
  for (const WaylandOutput& output : outputs) {
    if (output.x != x || output.y != y) continue;
    if (found != nullptr) return {};  // deux sorties au même endroit (miroir) : laquelle, inconnu
    found = &output;
  }
  return found != nullptr ? found->name : std::string();
}

std::string with_connector(std::string_view sdl_name, std::string_view connector) {
  if (connector.empty()) return std::string(sdl_name);
  return std::format("{} ({})", sdl_name, connector);
}

#ifdef AJP_WITH_WAYLAND_PRESENTATION

namespace {

struct Bound {
  wl_output* proxy = nullptr;
  WaylandOutput facts;
};

struct Listing {
  std::vector<std::unique_ptr<Bound>> outputs;
};

void on_geometry(void* data, wl_output* /*output*/, std::int32_t x, std::int32_t y, std::int32_t /*mm_w*/,
                 std::int32_t /*mm_h*/, std::int32_t /*subpixel*/, const char* /*make*/,
                 const char* /*model*/, std::int32_t /*transform*/) {
  auto* bound = static_cast<Bound*>(data);
  bound->facts.x = x;
  bound->facts.y = y;
}
void on_mode(void* /*data*/, wl_output* /*output*/, std::uint32_t /*flags*/, std::int32_t /*w*/,
             std::int32_t /*h*/, std::int32_t /*refresh*/) {}
void on_done(void* /*data*/, wl_output* /*output*/) {}
void on_scale(void* /*data*/, wl_output* /*output*/, std::int32_t /*factor*/) {}
void on_name(void* data, wl_output* /*output*/, const char* name) {
  static_cast<Bound*>(data)->facts.name = name != nullptr ? name : "";
}
void on_description(void* /*data*/, wl_output* /*output*/, const char* /*description*/) {}

constexpr wl_output_listener kOutput{.geometry = on_geometry,
                                     .mode = on_mode,
                                     .done = on_done,
                                     .scale = on_scale,
                                     .name = on_name,
                                     .description = on_description};

void on_global(void* data, wl_registry* registry, std::uint32_t name, const char* interface,
               std::uint32_t version) {
  if (std::strcmp(interface, wl_output_interface.name) != 0 || version < 4) return;
  auto* listing = static_cast<Listing*>(data);
  auto bound = std::make_unique<Bound>();
  bound->proxy = static_cast<wl_output*>(wl_registry_bind(registry, name, &wl_output_interface, 4));
  if (bound->proxy == nullptr) return;
  wl_output_add_listener(bound->proxy, &kOutput, bound.get());
  listing->outputs.push_back(std::move(bound));
}
void on_global_remove(void* /*data*/, wl_registry* /*registry*/, std::uint32_t /*name*/) {}

constexpr wl_registry_listener kRegistry{.global = on_global, .global_remove = on_global_remove};

// Toutes les sorties v4 : un aller-retour pour les globaux, un second pour les évènements des sorties liées.
std::vector<WaylandOutput> list_outputs(wl_display* display) {
  std::vector<WaylandOutput> out;
  wl_event_queue* queue = wl_display_create_queue(display);
  auto* wrapper = static_cast<wl_display*>(wl_proxy_create_wrapper(display));
  if (queue != nullptr && wrapper != nullptr) {
    wl_proxy_set_queue(reinterpret_cast<wl_proxy*>(wrapper), queue);
    Listing listing;
    wl_registry* registry = wl_display_get_registry(wrapper);
    wl_registry_add_listener(registry, &kRegistry, &listing);
    const bool read =
        wl_display_roundtrip_queue(display, queue) >= 0 && wl_display_roundtrip_queue(display, queue) >= 0;
    for (const auto& bound : listing.outputs) {
      if (read) out.push_back(bound->facts);
      wl_output_release(bound->proxy);
    }
    wl_registry_destroy(registry);
  }
  if (wrapper != nullptr) wl_proxy_wrapper_destroy(wrapper);
  if (queue != nullptr) wl_event_queue_destroy(queue);
  return out;
}

struct Known {
  int x = 0;
  int y = 0;
  std::string connector;
};

}  // namespace

std::string wayland_connector(SDL_DisplayID display) {
  const char* driver = SDL_GetCurrentVideoDriver();
  if (display == 0 || driver == nullptr || std::strcmp(driver, "wayland") != 0) return {};
  SDL_Rect bounds{};
  if (!SDL_GetDisplayBounds(display, &bounds)) return {};
  static std::mutex mutex;
  static std::map<SDL_DisplayID, Known> known;
  const std::scoped_lock lock(mutex);
  if (const auto it = known.find(display);
      it != known.end() && it->second.x == bounds.x && it->second.y == bounds.y)
    return it->second.connector;
  auto* wl = static_cast<wl_display*>(SDL_GetPointerProperty(
      SDL_GetGlobalProperties(), SDL_PROP_GLOBAL_VIDEO_WAYLAND_WL_DISPLAY_POINTER, nullptr));
  if (wl == nullptr) return {};
  std::string connector = output_at(list_outputs(wl), bounds.x, bounds.y);
  known[display] = Known{.x = bounds.x, .y = bounds.y, .connector = connector};
  return connector;
}

#else

std::string wayland_connector(SDL_DisplayID /*display*/) {
  return {};
}

#endif

}  // namespace ajp::app::play
