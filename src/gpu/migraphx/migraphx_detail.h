// Outils partagés par migraphx_runtime.cpp et migraphx_device.cpp.
#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <migraphx/migraphx.hpp>
#include <optional>
#include <string>

#include "gpu/vulkan/backend/net_runtime.h"

namespace ajp::gpu::migraphx_rt::detail {

std::unexpected<Refusal> refuse(RefusalCode code, std::string text);
std::optional<TensorType> type_of(migraphx_shape_datatype_t type);

// Le réseau sur la mémoire de l'appareil (interop) : programme compilé sans offload_copy, ses
// entrées et sorties liées aux tenseurs Vulkan importés, sans copie. run_ordered le met en file
// sur un flux HIP entre l'attente et le signal de la timeline Vulkan importée.
std::unique_ptr<vk::net::RuntimeNet> device_net(int device, migraphx::program program,
                                                vk::net::IoNames names);

// Importe une mémoire Vulkan exportée (descripteur opaque, allocation dédiée) ; HIP prend le
// descripteur en cas de succès. Refus DeviceUnavailable : ceux de HIP.
std::expected<std::unique_ptr<vk::net::ImportedMemory>, Refusal> import_memory(int device, int fd,
                                                                               std::uint64_t bytes);

// Importe le sémaphore timeline de la file Vulkan (descripteur opaque) ; HIP prend le descripteur
// en cas de succès. Les réseaux de device_net s'en servent dans run_ordered. Refus
// DeviceUnavailable : ceux de HIP.
std::expected<std::unique_ptr<vk::net::ImportedSemaphore>, Refusal> import_semaphore(int device, int fd);

}  // namespace ajp::gpu::migraphx_rt::detail
