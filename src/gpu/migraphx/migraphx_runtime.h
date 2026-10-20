// MIGraphX (AMD ROCm) comme runtime des réseaux du backend Vulkan (net_runtime.h) : programme
// compilé pour la cible « gpu » avec copies hôte intégrées (offload_copy), mis en cache en .mxr.
#pragma once

#include "gpu/vulkan/backend/net_runtime.h"

namespace ajp::gpu::migraphx_rt {

// Refus DeviceMismatch : appareil qui n'est pas AMD ; DeviceUnavailable : aucun GPU HIP.
std::expected<std::shared_ptr<vk::net::Runtime>, Refusal> make_runtime(const vk::DeviceCaps& caps);

}  // namespace ajp::gpu::migraphx_rt
