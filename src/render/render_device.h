// RenderDevice : l'appareil Vulkan de libplacebo, partagé par tout le lecteur.
//
// render crée l'instance (pl_vk_inst) avec les extensions que donne l'appelant (celles de SDL
// pour une fenêtre, aucune sans écran), puis l'appareil (pl_vulkan_create) avec ce que demandent
// le décodage Vulkan de FFmpeg (comme mpv, video/out/vulkan/context.c) et notre moteur : FP16,
// stockage 16 bits, unités matricielles si présentes, mémoire et sémaphores externes pour
// l'interop CUDA. Il décrit le tout par GpuDevice (core/gpu_device.h) : media, gpu et les
// backends partagent ainsi le même VkDevice, et soumettent sur ses files entre lock_queue et
// unlock_queue (le verrou de pl_vulkan).
//
// Thread main : l'objet se crée, s'utilise et se détruit sur le même thread. Les verrous de
// files, eux, se prennent depuis n'importe quel thread.
#pragma once

#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/gpu_device.h"
#include "core/types.h"

// Types de libplacebo, déclarés seulement : render les garde privés.
struct pl_gpu_t;
struct pl_vulkan_t;
struct pl_log_t;

namespace ajp::render {

// Crée la surface de la fenêtre sur l'instance (VkInstance) : un VkSurfaceKHR, 0 en cas d'échec.
// present la fournit par SDL_Vulkan_CreateSurface ; render la détruit avec l'instance.
using SurfaceFactory = std::function<std::uint64_t(void* instance)>;

struct RenderDeviceOptions {
  // Extensions d'instance exigées (SDL_Vulkan_GetInstanceExtensions) ; vide sans écran.
  std::vector<std::string> instance_extensions;
  // Vide : sans écran (export, tests). Sinon appelée une fois, entre l'instance et l'appareil.
  SurfaceFactory make_surface;
  // PFN_vkGetInstanceProcAddr de l'appelant (SDL_Vulkan_GetVkGetInstanceProcAddr) ; nullptr :
  // celui du chargeur Vulkan lié à libplacebo.
  VoidFunction get_instance_proc_addr = nullptr;
  // Appareil imposé (UUID) ; absent : le premier qui convient, comme libplacebo le choisit.
  std::optional<DeviceUuid> device_uuid;
  // Accepte un appareil logiciel (lavapipe) : tests et CI seulement.
  bool allow_software = false;
};

class RenderDevice {
 public:
  // DeviceUnavailable : aucun appareil ne convient ; FeatureUnavailable : fonction obligatoire
  // absente ; VulkanCallFailed : instance ou surface refusées.
  static std::expected<RenderDevice, Refusal> create(const RenderDeviceOptions& options);

  RenderDevice(RenderDevice&&) noexcept;
  RenderDevice& operator=(RenderDevice&&) noexcept;
  RenderDevice(const RenderDevice&) = delete;
  RenderDevice& operator=(const RenderDevice&) = delete;
  ~RenderDevice();

  // La description partagée ; ses pointeurs (instance, appareil, chaîne de fonctionnalités,
  // contexte du verrou) restent valides tant que cet objet vit.
  const GpuDevice& description() const;

  const pl_gpu_t* gpu() const;
  const pl_vulkan_t* vulkan() const;
  const pl_log_t* log() const;
  std::uint64_t surface() const;  // VkSurfaceKHR ; 0 sans écran

 private:
  struct Impl;
  explicit RenderDevice(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace ajp::render
