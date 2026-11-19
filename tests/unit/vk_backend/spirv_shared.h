// La mémoire partagée que déclare un module SPIR-V embarqué, ses constantes de spécialisation données : la
// somme des tailles de ses variables Workgroup, longueurs de tableaux évaluées comme le ferait le pilote
// (OpSpecConstantOp sur les constantes de spécialisation). Sert à comparer le compte de l'hôte
// (matrix_shared_bytes et ses voisins, net_lower_detail.h, net_conv_m8.h) à ce que le noyau déclare vraiment.
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <span>

namespace kirari::gpu::vk::test {

// constants : SpecId → valeur, la valeur par défaut du module pour les autres. nullopt : module illisible, ou
// opération de constante que l'évaluateur ne connaît pas.
std::optional<std::uint64_t> spirv_shared_bytes(std::span<const std::uint32_t> words,
                                                const std::map<std::uint32_t, std::uint32_t>& constants);

// Les constantes d'un dispatch (Dispatch::constants : SpecId i pour l'entrée i), comme le pipeline les passe.
std::map<std::uint32_t, std::uint32_t> spec_constants(std::span<const std::uint32_t> constants);

}  // namespace kirari::gpu::vk::test
