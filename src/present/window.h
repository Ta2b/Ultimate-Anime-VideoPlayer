// Fenêtre SDL3 du lecteur : Vulkan, haute densité, plein écran sans changer le mode de l'écran
// (le plein écran du bureau ; sous Wayland, le lecteur ne change jamais le mode).
// Thread main seulement, comme toute la vidéo SDL.
#pragma once

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include "core/profile.h"
#include "core/types.h"

struct SDL_Window;

namespace ajp::present {

struct WindowOptions {
  std::string title = "Ultimate Anime VideoPlayer";
  int width = 1280;  // taille logique, en points ; les pixels suivent la densité de l'écran
  int height = 720;
  bool fullscreen = false;
  bool resizable = true;
};

namespace window_limits {
inline constexpr core::Range<int> kSide{1, 16384};
}  // namespace window_limits

struct PixelSize {
  int width = 0;
  int height = 0;
  friend bool operator==(const PixelSize&, const PixelSize&) = default;
};

class Window {
 public:
  // Démarre la vidéo de SDL3 (compte de références : SDL_InitSubSystem) et ouvre la fenêtre.
  // Refuse une taille hors de [1, 16384] ; PresentFailed si SDL3 refuse.
  static std::expected<Window, Refusal> create(const WindowOptions& options);

  Window(Window&& other) noexcept;
  Window& operator=(Window&& other) noexcept;
  Window(const Window&) = delete;
  Window& operator=(const Window&) = delete;
  ~Window();

  SDL_Window* sdl() const { return window_; }
  std::uint32_t id() const;  // SDL_WindowID, pour trier les événements

  // Extensions d'instance Vulkan qu'exige la surface de cette fenêtre, à passer à render.
  std::expected<std::vector<std::string>, Refusal> vulkan_instance_extensions() const;

  PixelSize pixel_size() const;
  // Ni cachée, ni réduite, ni masquée : le compositeur la présente. Masquée comprend l'écran en
  // veille sous Wayland (xdg_toplevel « suspended »). À jour une fois les événements SDL lus.
  bool presentable() const { return presentable(window_); }
  static bool presentable(SDL_Window* window);
  std::expected<void, Refusal> set_fullscreen(bool fullscreen);
  std::expected<void, Refusal> set_size(int width, int height);  // en points

 private:
  explicit Window(SDL_Window* window) : window_(window) {}
  void close();

  SDL_Window* window_ = nullptr;
};

}  // namespace ajp::present
