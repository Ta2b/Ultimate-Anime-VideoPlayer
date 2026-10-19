// Description d'un appareil pl_vulkan en GpuDevice (core/gpu_device.h). Interne à render.
#pragma once

#include <expected>

#include <libplacebo/vulkan.h>

#include "core/gpu_device.h"
#include "core/types.h"

namespace ajp::render::detail {

// Le verrou des files : lock_queue et unlock_queue de GpuDevice, avec leur contexte.
struct QueueLocks {
  QueueLock lock = nullptr;
  QueueLock unlock = nullptr;
  void* context = nullptr;
};

// Identité, pilote, extensions, fonctionnalités et files de vk. Les familles retenues sont
// celles où libplacebo a créé des files : graphique, calcul, transfert et décodage vidéo.
std::expected<GpuDevice, Refusal> describe_device(pl_vk_inst instance, pl_vulkan vk, const QueueLocks& locks);

}  // namespace ajp::render::detail
