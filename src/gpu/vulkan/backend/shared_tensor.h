// Tenseurs partagés avec un runtime (interop) : un VkBuffer de stockage dans sa propre mémoire
// d'appareil, exportée sous le descripteur du runtime et importée par lui : descripteur opaque
// pour HIP de MIGraphX (Linux), dma-buf ou HANDLE NT pour OpenCL d'OpenVINO sur une carte Arc
// (Linux, Windows). La chaîne Vulkan et le réseau travaillent dans la même mémoire, sans copie par
// l'hôte ; la timeline de la file s'exporte de même (Linux), pour que le runtime s'y ordonne côté
// GPU.
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
#include "gpu/vulkan/compute/memory.h"

namespace ajp::gpu::vk {

// L'appareil exporte-t-il un tampon de stockage sous ce descripteur (extensions activées :
// VK_KHR_external_memory_fd, plus VK_EXT_external_memory_dma_buf pour le dma-buf, sous Linux ;
// VK_KHR_external_memory_win32 sous Windows) ?
bool exports_memory(const VkContext& context, net::MemoryHandle handle);

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

// Un tenseur de bytes octets, exporté sous Runtime::memory_handle et importé par le runtime,
// inscrit dans views (sa vue du runtime) tant qu'il vit. verify : la chaîne et le runtime y
// échangent d'abord deux mots (ImportedMemory::exchange), pour vérifier qu'ils voient bien la même
// mémoire ; memory sert au tampon de relecture. Refus : ceux de Vulkan, de l'import ;
// FeatureUnavailable : descripteur que l'appareil n'exporte pas, runtime qui voit une autre
// mémoire.
std::expected<std::unique_ptr<TensorBuffer>, Refusal> shared_tensor(
    VkContext& context, MemoryAllocator& memory, net::Runtime& runtime, TensorType type, const Shape& shape,
    std::uint64_t bytes, std::shared_ptr<HostViews> views, bool verify);

}  // namespace ajp::gpu::vk
