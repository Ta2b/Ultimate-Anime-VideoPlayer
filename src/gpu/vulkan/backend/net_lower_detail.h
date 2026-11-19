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

namespace kirari::gpu::vk::net::detail {

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
// Variante à halo (HALO, B jamais partagée) : deux fenêtres de hpix pixels au lieu des deux
// tampons de A, même pas de ligne.
std::uint32_t matrix_halo_shared_bytes(std::uint32_t subgroups, std::uint32_t kc, std::uint32_t hpix);
// Pixels de la fenêtre d'un groupe à halo : (lignes + halo) × (16 + halo).
std::uint32_t matrix_halo_pixels(std::uint32_t rows, std::uint32_t kh, std::uint32_t kw, std::uint32_t dh,
                                 std::uint32_t dw);

// Épilogue des constantes 0 à 29 communes aux shaders de convolution (net_conv.comp).
struct ConvConstants {
  std::uint32_t tr = 0;
  std::uint32_t lr = 0;
  std::uint32_t crp = 1;
  std::uint32_t act = 0;
  float alpha = 0.0F;
  float lo = 0.0F;
  float hi = 0.0F;
  std::uint32_t residual = 0;  // 1 : ajouté avant l'activation ; 2 : porte (Epilogue::gate)
};
// Les constantes 0 à 29, dans l'ordre des constant_id de net_conv.comp et net_conv_coopmat.glsl.
std::vector<std::uint32_t> conv_constants(const Desc4& x, std::uint32_t tx, const Desc4& y, std::uint32_t ty,
                                          const Window& w, std::uint32_t groups, const ConvConstants& e);
// La forme calculée par une Conv avant son DepthToSpace fondu : le rangement (disposition, canaux rangés)
// reste celui de la sortie y.
Desc4 result_desc(const Desc4& y, const ir::Epilogue& e);
// Constantes du DepthToSpace fondu : taille des blocs (0 sans), ordre CRD.
std::array<std::uint32_t, 2> space_constants(const ir::Epilogue& e);

}  // namespace kirari::gpu::vk::net::detail
