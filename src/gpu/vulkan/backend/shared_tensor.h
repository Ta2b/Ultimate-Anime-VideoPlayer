// Tenseurs partagés avec un runtime (interop, Linux) : un VkBuffer de stockage dans sa propre
// mémoire d'appareil, exportée en descripteur opaque et importée par le runtime (MIGraphX par
// HIP). La chaîne Vulkan et le réseau travaillent dans la même mémoire, sans copie par l'hôte ;
// la timeline de la file s'exporte de même, pour que le runtime s'y ordonne côté GPU.
//
// Sur un GPU intégré (tous systèmes), le sens inverse : le runtime alloue une mémoire hôte qu'il
// lit sans copie (USM d'OpenVINO) et Vulkan l'importe (VK_EXT_external_memory_host).
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

// L'alignement des mémoires hôte que l'appareil importe (VK_EXT_external_memory_host activée,
// minImportedHostPointerAlignment), ou 0 s'il n'en importe pas. Tous systèmes.
std::uint64_t host_import_alignment(const VkContext& context);

// Un tenseur de bytes octets dans une mémoire hôte du runtime (Runtime::allocate_host), importée
// par Vulkan : vu par l'hôte et le runtime à la même adresse, inscrit dans views tant qu'il vit.
// verify : le GPU écrit et relit d'abord quelques octets, pour vérifier qu'il voit bien cette
// mémoire et pas une copie (une soumission attendue sur l'hôte).
// Refus : ceux du runtime (FeatureUnavailable s'il n'en alloue pas, FormatUnsupported pas pour ce
// tenseur), de Vulkan ; FeatureUnavailable : le GPU n'en voit qu'une copie.
std::expected<std::unique_ptr<TensorBuffer>, Refusal> host_imported_tensor(
    VkContext& context, net::Runtime& runtime, TensorType type, const Shape& shape, std::uint64_t bytes,
    std::uint64_t alignment, std::shared_ptr<HostViews> views, bool verify);

// Un tenseur de bytes octets, inscrit dans views (adresse du runtime) tant qu'il vit. Refus :
// ceux de Vulkan, de l'import, FeatureUnavailable hors de Linux.
std::expected<std::unique_ptr<TensorBuffer>, Refusal> shared_tensor(VkContext& context, net::Runtime& runtime,
                                                                    TensorType type, const Shape& shape,
                                                                    std::uint64_t bytes,
                                                                    std::shared_ptr<HostViews> views);

}  // namespace ajp::gpu::vk
