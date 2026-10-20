// Outils communs aux fichiers de l'abaissement (interne au module).
#pragma once

#include <cstdint>
#include <vector>

#include "gpu/vulkan/backend/net_params.h"

namespace ajp::gpu::vk::net::detail {

// TDesc de net_tensor.glsl : n, c, h, w, ordre, canaux rangés.
void put(std::vector<std::uint32_t>& push, const Desc4& d);
void put_f(std::vector<std::uint32_t>& push, float value);

}  // namespace ajp::gpu::vk::net::detail
