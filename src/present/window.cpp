#include "present/window.h"

#include <format>
#include <utility>

#include <SDL3/SDL_init.h>
#include <SDL3/SDL_video.h>
#include <SDL3/SDL_vulkan.h>

namespace ajp::present {

namespace {

std::unexpected<Refusal> sdl_refusal(std::string_view call) {
  return std::unexpected(Refusal{RefusalCode::PresentFailed, std::format("{} : {}", call, SDL_GetError())});
}

std::expected<void, Refusal> check_size(int width, int height) {
  if (!window_limits::kSide.contains(width) || !window_limits::kSide.contains(height)) {
    return std::unexpected(Refusal{RefusalCode::FieldOutOfRange,
                                   std::format("fenêtre {}x{} hors de [1, 16384]", width, height)});
  }
  return {};
}

}  // namespace

std::expected<Window, Refusal> Window::create(const WindowOptions& options) {
  if (auto valid = check_size(options.width, options.height); !valid) return std::unexpected(valid.error());
  if (!SDL_InitSubSystem(SDL_INIT_VIDEO)) return sdl_refusal("SDL_InitSubSystem(VIDEO)");
  SDL_WindowFlags flags = SDL_WINDOW_VULKAN | SDL_WINDOW_HIGH_PIXEL_DENSITY;
  if (options.resizable) flags |= SDL_WINDOW_RESIZABLE;
  if (options.fullscreen) flags |= SDL_WINDOW_FULLSCREEN;
  SDL_Window* window = SDL_CreateWindow(options.title.c_str(), options.width, options.height, flags);
  if (window == nullptr) {
    auto refusal = sdl_refusal("SDL_CreateWindow");
    SDL_QuitSubSystem(SDL_INIT_VIDEO);
    return refusal;
  }
  return Window(window);
}

Window::Window(Window&& other) noexcept : window_(std::exchange(other.window_, nullptr)) {}

Window& Window::operator=(Window&& other) noexcept {
  if (this != &other) {
    close();
    window_ = std::exchange(other.window_, nullptr);
  }
  return *this;
}

Window::~Window() {
  close();
}

void Window::close() {
  if (window_ == nullptr) return;
  SDL_DestroyWindow(window_);
  SDL_QuitSubSystem(SDL_INIT_VIDEO);
  window_ = nullptr;
}

std::uint32_t Window::id() const {
  return SDL_GetWindowID(window_);
}

std::expected<std::vector<std::string>, Refusal> Window::vulkan_instance_extensions() const {
  Uint32 count = 0;
  const char* const* names = SDL_Vulkan_GetInstanceExtensions(&count);
  if (names == nullptr) return sdl_refusal("SDL_Vulkan_GetInstanceExtensions");
  std::vector<std::string> extensions;
  extensions.reserve(count);
  for (Uint32 i = 0; i < count; ++i) extensions.emplace_back(names[i]);
  return extensions;
}

std::expected<PixelSize, Refusal> Window::pixel_size() const {
  PixelSize size;
  if (!SDL_GetWindowSizeInPixels(window_, &size.width, &size.height)) {
    return sdl_refusal("SDL_GetWindowSizeInPixels");
  }
  return size;
}

bool Window::presentable(SDL_Window* window) {
  const SDL_WindowFlags hidden = SDL_WINDOW_HIDDEN | SDL_WINDOW_MINIMIZED | SDL_WINDOW_OCCLUDED;
  return (SDL_GetWindowFlags(window) & hidden) == 0;
}

std::expected<void, Refusal> Window::set_fullscreen(bool fullscreen) {
  if (!SDL_SetWindowFullscreen(window_, fullscreen)) return sdl_refusal("SDL_SetWindowFullscreen");
  return {};
}

std::expected<void, Refusal> Window::set_size(int width, int height) {
  if (auto valid = check_size(width, height); !valid) return valid;
  if (!SDL_SetWindowSize(window_, width, height)) return sdl_refusal("SDL_SetWindowSize");
  return {};
}

}  // namespace ajp::present
