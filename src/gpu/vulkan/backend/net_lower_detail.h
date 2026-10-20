// Outils communs aux fichiers de l'abaissement (interne au module).
#pragma once

#include <cstdint>
#include <vector>

#include "gpu/vulkan/backend/net_params.h"

namespace ajp::gpu::vk::net::detail {

// TDesc de net_tensor.glsl : n, c, h, w, ordre, canaux rangés.
void put(std::vector<std::uint32_t>& push, const Desc4& d);
void put_f(std::vector<std::uint32_t>& push, float value);

// Mémoire partagée que déclare la variante coopmat (net_conv_coopmat.glsl) : deux tampons de A
// (lignes de kc / 8 uvec4, plus un dès kc = 32), la tuile de l'épilogue et deux tampons de B, de
// kc lignes de np canaux de sortie (arrondis à 16) quand B passe par la mémoire partagée (np > 0),
// d'un uvec4 sinon, que le noyau déclare quand même. Le compte doit égaler la déclaration : 32
// octets oubliés suffisent à dépasser maxComputeSharedMemorySize et à perdre l'appareil.
std::uint32_t matrix_shared_bytes(std::uint32_t subgroups, std::uint32_t tm, std::uint32_t kc,
                                  std::uint32_t np);

}  // namespace ajp::gpu::vk::net::detail
