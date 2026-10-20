// Tenseurs partagés avec un runtime (interop, Linux) : un VkBuffer de stockage dans sa propre
// mémoire d'appareil, exportée en descripteur opaque et importée par le runtime (MIGraphX par
// HIP). La chaîne Vulkan et le réseau travaillent dans la même mémoire, sans copie par l'hôte ;
// la timeline de la file s'exporte de même, pour que le runtime s'y ordonne côté GPU.
#pragma once

#include <cstdint>
#include <expected>
#include <memory>

#include "core/types.h"
#include "gpu/tensor.h"
#include "gpu/vulkan/backend/net_runtime.h"
#include "gpu/vulkan/backend/vulkan_backend_detail.h"
#include "gpu/vulkan/compute/context.h"

namespace ajp::gpu::vk {

// L'appareil exporte-t-il sa mémoire en descripteur opaque (VK_KHR_external_memory_fd activée) ?
// Toujours faux sous Windows.
bool exports_memory(const VkContext& context);

// L'appareil exporte-t-il ses sémaphores timeline en descripteur opaque
// (VK_KHR_external_semaphore_fd activée, export permis) ? Toujours faux sous Windows.
bool exports_semaphores(const VkContext& context);

// Le descripteur opaque d'un sémaphore créé exportable, que l'appelant donne au runtime ou ferme.
// Refus : ceux de Vulkan, FeatureUnavailable hors de Linux.
std::expected<int, Refusal> semaphore_fd(const VkContext& context, VkSemaphore semaphore);

// Un tenseur de bytes octets, inscrit dans views (adresse du runtime) tant qu'il vit. Refus :
// ceux de Vulkan, de l'import, FeatureUnavailable hors de Linux.
std::expected<std::unique_ptr<TensorBuffer>, Refusal> shared_tensor(const VkContext& context,
                                                                    net::Runtime& runtime, TensorType type,
                                                                    const Shape& shape, std::uint64_t bytes,
                                                                    std::shared_ptr<HostViews> views);

}  // namespace ajp::gpu::vk
