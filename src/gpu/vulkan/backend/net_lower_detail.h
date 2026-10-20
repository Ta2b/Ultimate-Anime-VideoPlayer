// Outils communs aux fichiers de l'abaissement (interne au module).
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <string_view>
#include <vector>

#include "core/types.h"
#include "gpu/ir/graph.h"
#include "gpu/vulkan/backend/net_params.h"

namespace ajp::gpu::vk::net::detail {

// TDesc de net_tensor.glsl : n, c, h, w, ordre, canaux rangés.
void put(std::vector<std::uint32_t>& push, const Desc4& d);
void put_f(std::vector<std::uint32_t>& push, float value);

// Un refus nommé par l'opérateur et le nœud.
std::unexpected<Refusal> refuse_node(RefusalCode code, const ir::Node& node, std::string_view detail);
std::uint32_t count_of(const Desc4& d);
std::array<std::uint32_t, 4> dims_of(const Desc4& d);

constexpr std::uint32_t kIdentityPerm = 0U | (1U << 2U) | (2U << 4U) | (3U << 6U);

// Bloc push de net_copy.comp : vues, boîte parcourue, décalage de sortie, départ, pas, perm.
struct CopyPush {
  std::array<std::uint32_t, 4> iter{};
  std::array<std::uint32_t, 4> out_offset{};
  std::array<std::int32_t, 4> start{};
  std::array<std::int32_t, 4> step{1, 1, 1, 1};
  std::uint32_t perm = kIdentityPerm;
};

// Le bloc push de net_copy.comp : les deux vues puis le bloc.
std::vector<std::uint32_t> copy_push(const Desc4& x, const Desc4& y, const CopyPush& c);

// Mémoire partagée que déclare la variante coopmat (net_conv_coopmat.glsl) : deux tampons de A
// (lignes de kc / 8 uvec4, plus un dès kc = 32), la tuile de l'épilogue et deux tampons de B, de
// kc lignes de np canaux de sortie (arrondis à 16) quand B passe par la mémoire partagée (np > 0),
// d'un uvec4 sinon, que le noyau déclare quand même. Le compte doit égaler la déclaration : 32
// octets oubliés suffisent à dépasser maxComputeSharedMemorySize et à perdre l'appareil.
std::uint32_t matrix_shared_bytes(std::uint32_t subgroups, std::uint32_t tm, std::uint32_t kc,
                                  std::uint32_t np);

}  // namespace ajp::gpu::vk::net::detail
