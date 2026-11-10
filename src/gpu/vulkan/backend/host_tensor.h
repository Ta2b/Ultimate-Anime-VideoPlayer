// Les tenseurs d'un runtime dans sa propre mémoire hôte, importée par Vulkan (VK_EXT_external_memory_host) :
// sur un GPU intégré, le runtime (Core ML, IOSurface) les lit et les écrit sans copie.
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

}  // namespace ajp::gpu::vk
