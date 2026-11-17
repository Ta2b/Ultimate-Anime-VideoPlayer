// Les variantes coopmat à produits de faible précision d'une Conv quantifiée (ConvAttrs::in_quant et
// quant_grid) : FP8 E4M3 (tuiles e4m3 des RDNA 4, Target::matrix_f8, net_conv_coopmat_fp8.glsl) et INT8
// (tuiles s8 → s32 des RDNA 3 et 4, Target::matrix_i8, net_conv_coopmat_int8.glsl). Interne à l'abaissement
// (net_lower_conv.cpp) : éligibilité, candidats de l'autotuner, poids et échelles rangés.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "gpu/ir/graph.h"
#include "gpu/vulkan/backend/net_program.h"

namespace ajp::gpu::vk::net::detail {

inline constexpr std::string_view kMatrixFp8 = "coopmat_fp8";    // produits E4M3 × E4M3, sommes en float
inline constexpr std::string_view kMatrixInt8 = "coopmat_int8";  // produits INT8 × INT8, sommes entières

// La grille d'une variante à faible précision, nullopt pour les autres.
std::optional<ir::QuantGrid> low_bit_grid(std::string_view variant);
// Le shader de la variante de cette grille.
std::string_view low_bit_shader(ir::QuantGrid grid);

// L'échelle d'un arrondi telle que la lit net_quant (net_activation.glsl) : positive sur la grille E4M3,
// négative sur la grille INT8 (ConvAttrs::quant_grid), nulle sans arrondi.
float grid_scale(const ir::ConvAttrs& attrs, float scale);

// Taille de sous-groupe des matrices coopératives : 32 exigé quand l'appareil le permet (les WMMA des RDNA en
// wave32, comme llama.cpp ; le pilote AMD annonce 64 par défaut), sinon celle du pilote.
std::uint32_t matrix_subgroup(const Target& target);

// La variante de cette grille convient-elle au nœud ? Tuiles de la grille annoncées par le pilote, shader
// compilé, Conv à groupe unique dont l'entrée float16 en Nhwc16 est arrondie à cette grille, au plus 16
// tuiles de 16 canaux de sortie, poids constants rangés sur la grille (recover_fp8_weights,
// recover_int8_weights).
bool low_bit_eligible(const ir::Graph& graph, const ir::Node& node, const Target& target, ir::QuantGrid grid);

// Les candidats des variantes éligibles, ajoutés à out : derrière le noyau direct, qui reste le choix par
// défaut et la référence de l'autotuner (sortie comparée avant de retenir la variante).
void add_low_bit_candidates(const ir::Graph& graph, const ir::Node& node, const Target& target,
                            std::vector<KernelChoice>& out);

// Les poids rangés [ky][kx][cin arrondi][cout arrondi] en octets de la grille, puis les biais et les échelles
// (in_scale × échelle de chaque canal) ; nullopt si les poids ne tombent pas sur la grille.
struct LowBitBlobs {
  std::vector<std::byte> weights;
  std::vector<std::byte> bias;
};
std::optional<LowBitBlobs> low_bit_blobs(const ir::Tensor& weights, const ir::Tensor* bias,
                                         ir::QuantGrid grid, float in_scale, std::uint32_t cin_padded,
                                         std::uint32_t cout_padded);

}  // namespace ajp::gpu::vk::net::detail
