#include "present/surface.h"

#include <format>
#include <utility>

#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>

#include "present/vk_handle.h"

namespace ajp::present {

std::expected<Surface, Refusal> Surface::create(const Window& window, void* vk_instance) {
  if (vk_instance == nullptr) {
    return std::unexpected(
        Refusal{RefusalCode::PresentFailed, "SDL_Vulkan_CreateSurface : instance Vulkan nulle"});
  }
  auto* instance = static_cast<VkInstance>(vk_instance);
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  if (!SDL_Vulkan_CreateSurface(window.sdl(), instance, nullptr, &surface)) {
    return std::unexpected(
        Refusal{RefusalCode::PresentFailed, std::format("SDL_Vulkan_CreateSurface : {}", SDL_GetError())});
  }
  return Surface(vk_instance, to_u64(surface));
}

Surface::Surface(Surface&& other) noexcept
    : instance_(std::exchange(other.instance_, nullptr)), surface_(std::exchange(other.surface_, 0)) {}

Surface& Surface::operator=(Surface&& other) noexcept {
  if (this != &other) {
    destroy();
    instance_ = std::exchange(other.instance_, nullptr);
    surface_ = std::exchange(other.surface_, 0);
  }
  return *this;
}

Surface::~Surface() {
  destroy();
}

void Surface::destroy() {
  if (surface_ == 0) return;
  SDL_Vulkan_DestroySurface(static_cast<VkInstance>(instance_), from_u64<VkSurfaceKHR>(surface_), nullptr);
  surface_ = 0;
  instance_ = nullptr;
}

}  // namespace ajp::present
