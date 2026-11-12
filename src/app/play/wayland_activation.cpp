#include "app/play/wayland_activation.h"

#ifdef AJP_WITH_WAYLAND_PRESENTATION
#include <cstdint>
#include <cstring>
#include <string>
#include <wayland-client.h>

#include <SDL3/SDL_properties.h>
#include <SDL3/SDL_video.h>

#include "xdg-activation-v1-client.h"
#endif

namespace ajp::app::play {

#ifdef AJP_WITH_WAYLAND_PRESENTATION

namespace {

void on_global(void* data, wl_registry* registry, std::uint32_t name, const char* interface,
               std::uint32_t /*version*/) {
  if (std::strcmp(interface, xdg_activation_v1_interface.name) != 0) return;
  *static_cast<xdg_activation_v1**>(data) =
      static_cast<xdg_activation_v1*>(wl_registry_bind(registry, name, &xdg_activation_v1_interface, 1));
}

void on_global_remove(void* /*data*/, wl_registry* /*registry*/, std::uint32_t /*name*/) {}

constexpr wl_registry_listener kRegistry{.global = on_global, .global_remove = on_global_remove};

}  // namespace

bool wayland_activate(SDL_Window* window, std::string_view token) {
  if (window == nullptr || token.empty()) return false;
  const char* driver = SDL_GetCurrentVideoDriver();
  if (driver == nullptr || std::strcmp(driver, "wayland") != 0) return false;
  const SDL_PropertiesID props = SDL_GetWindowProperties(window);
  auto* display = static_cast<wl_display*>(
      SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WAYLAND_DISPLAY_POINTER, nullptr));
  auto* surface = static_cast<wl_surface*>(
      SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER, nullptr));
  if (display == nullptr || surface == nullptr) return false;
  // Une file à nous, comme placebo_wayland.cpp : les événements du registre ne passent pas par la boucle de
  // SDL. Un aller-retour suffit (les globaux), puis la demande part avec le flush.
  wl_event_queue* queue = wl_display_create_queue(display);
  auto* wrapper = static_cast<wl_display*>(wl_proxy_create_wrapper(display));
  bool done = false;
  if (queue != nullptr && wrapper != nullptr) {
    wl_proxy_set_queue(reinterpret_cast<wl_proxy*>(wrapper), queue);
    wl_registry* registry = wl_display_get_registry(wrapper);
    xdg_activation_v1* activation = nullptr;
    wl_registry_add_listener(registry, &kRegistry, static_cast<void*>(&activation));
    const bool listed = wl_display_roundtrip_queue(display, queue) >= 0;
    if (activation != nullptr) {
      if (listed) {
        const std::string text(token);
        xdg_activation_v1_activate(activation, text.c_str(), surface);
      }
      xdg_activation_v1_destroy(activation);
      done = listed && wl_display_flush(display) >= 0;
    }
    wl_registry_destroy(registry);
  }
  if (wrapper != nullptr) wl_proxy_wrapper_destroy(wrapper);
  if (queue != nullptr) wl_event_queue_destroy(queue);
  return done;
}

#else

bool wayland_activate(SDL_Window* /*window*/, std::string_view /*token*/) {
  return false;
}

#endif

}  // namespace ajp::app::play
