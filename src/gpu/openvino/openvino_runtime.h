// OpenVINO (Intel) comme runtime des réseaux du backend Vulkan (net_runtime.h) : modèle compilé
// pour le GPU Intel de même UUID que l'appareil Vulkan, en latence et en f16, mis en cache par
// export_model ; entrées et sorties en F16 côté lecteur, converties sur l'appareil.
#pragma once

#include "gpu/vulkan/backend/net_runtime.h"

namespace ajp::gpu::openvino_rt {

// Refus DeviceMismatch : appareil qui n'est pas Intel ; DeviceUnavailable : aucun GPU OpenVINO.
std::expected<std::shared_ptr<vk::net::Runtime>, Refusal> make_runtime(const vk::DeviceCaps& caps);

}  // namespace ajp::gpu::openvino_rt
