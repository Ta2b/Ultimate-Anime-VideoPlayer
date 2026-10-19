// Surface Vulkan de la fenêtre, créée sur l'instance de render (SDL_Vulkan_CreateSurface).
//
// Ordre de construction, que app tient :
//   1. present ouvre la fenêtre et donne ses extensions d'instance à render ;
//   2. render crée l'instance (pl_vk_inst) avec ces extensions ;
//   3. present crée la surface sur cette instance ;
//   4. render crée l'appareil (pl_vulkan_params::surface : une file qui sait présenter,
//      VK_KHR_swapchain) ;
//   5. present crée la swapchain sur cet appareil.
// Destruction dans l'ordre inverse : swapchain, surface, appareil, instance, fenêtre.
#pragma once

#include <cstdint>
#include <expected>

#include "core/types.h"
#include "present/window.h"

namespace ajp::present {

class Surface {
 public:
  // vk_instance : le VkInstance de render (pl_vk_inst::instance), créé avec les extensions
  // de Window::vulkan_instance_extensions().
  static std::expected<Surface, Refusal> create(const Window& window, void* vk_instance);

  Surface(Surface&& other) noexcept;
  Surface& operator=(Surface&& other) noexcept;
  Surface(const Surface&) = delete;
  Surface& operator=(const Surface&) = delete;
  ~Surface();

  // VkSurfaceKHR, handle non dispatchable de 64 bits : pour pl_vulkan_params::surface.
  std::uint64_t handle() const { return surface_; }
  void* instance() const { return instance_; }

 private:
  Surface(void* instance, std::uint64_t surface) : instance_(instance), surface_(surface) {}
  void destroy();

  void* instance_ = nullptr;
  std::uint64_t surface_ = 0;
};

}  // namespace ajp::present
