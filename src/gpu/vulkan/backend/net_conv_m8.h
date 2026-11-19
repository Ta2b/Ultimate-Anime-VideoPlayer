// La variante coopmat_m8 d'une Conv float16 : matrices coopératives en tuiles 8 × 16 × 16 (f16 × f16 → f32,
// Target::matrix_m8), celles qu'annoncent les XMX d'Intel Xe2 (Lunar Lake, Battlemage) sous ANV. Noyau
// net_conv_coopmat_m8_f16 : net_conv_coopmat.glsl en tuiles de 8 pixels (KIRARI_COOPMAT_M8), poids rangés
// comme la variante coopmat (matrix_weights : N et K restent 16), B lue dans le tampon ou partagée, ni halo,
// ni épilogue en registres, ni canaux découpés. Sous-groupes de 16 exigés. Jamais en tête des candidats :
// écrite sans carte Intel pour l'essayer, elle n'est retenue que par l'autotuner, qui compare sa sortie à
// celle du noyau direct, puis par la porte de bout en bout. Interne à l'abaissement (net_lower_conv.cpp,
// net_lower.cpp).
#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "gpu/ir/graph.h"
#include "gpu/vulkan/backend/net_program.h"

namespace kirari::gpu::vk::net::detail {

inline constexpr std::string_view kMatrixM8 = "coopmat_m8";
inline constexpr std::string_view kMatrixM8Shader = "net_conv_coopmat_m8_f16";
inline constexpr std::uint32_t kMatrixM8Pixels = 8;  // M : pixels d'une tuile

// Taille de sous-groupe de la variante : 16, la largeur SIMD des Xe2 et celle de leur tuile N, quand
// l'appareil laisse l'exiger (sous-groupes pleins, entre subgroup_min et subgroup_max) ; 0 sinon, et la
// variante n'est pas proposée. Pas matrix_subgroup, qui exige 32 dès qu'il le peut (les WMMA des RDNA en
// wave32).
std::uint32_t matrix_m8_subgroup(const Target& target);

// Mémoire partagée que déclare net_conv_coopmat_m8_f16 : comme matrix_shared_bytes, en tuiles de 8 pixels
// (deux tampons de A de subgroups × tm × 8 lignes, une tuile de l'épilogue de 8 × 16 floats par sous-groupe,
// deux tampons de B). Le compte doit égaler la déclaration du noyau.
std::uint32_t matrix_m8_shared_bytes(std::uint32_t subgroups, std::uint32_t tm, std::uint32_t kc,
                                     std::uint32_t np);

// La variante convient-elle au nœud ? Tuiles annoncées, sous-groupes de 16 permis, Conv à groupe unique et
// sans arrondi FP8 ni INT8, entrée float16 en Nhwc16, au plus 16 tuiles de 16 canaux de sortie, poids
// constants.
bool matrix_m8_eligible(const ir::Graph& graph, const ir::Node& node, const Target& target);

// Les candidats de la variante, ajoutés à out (derrière le noyau direct) : tuning {tm, sous-groupes} ou
// {tm, sous-groupes, 1} pour B en mémoire partagée.
void add_matrix_m8_candidates(const ir::Graph& graph, const ir::Node& node, const Target& target,
                              std::vector<KernelChoice>& out);

}  // namespace kirari::gpu::vk::net::detail
