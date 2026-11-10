// Ce qu'un module SPIR-V promet de ses tampons de stockage : les bindings (jeu 0) que toutes ses variables
// déclarent readonly (décoration NonWritable sur la variable ou sur chaque membre de son bloc). Le Recorder
// ne pose une barrière entre deux dispatchs que si l'un écrit ce que l'autre touche : deux lectures d'un même
// tampon, ou du tampon factice d'un binding inutilisé, se recouvrent.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace ajp::gpu::vk {

// Un bit par binding de 0 à 63 ; 0 pour un module illisible (tout compte alors comme écrit).
std::uint64_t spirv_readonly_bindings(std::span<const std::uint32_t> words);

// Le module avec l'arrondi et les sous-normaux du float 16 bits déclarés (SPV_KHR_float_controls, A45-15) :
// RoundingModeRTE 16 si rte, DenormPreserve 16 si denorm, sur chaque point d'entrée. Sans ces modes, la
// spécification laisse le pilote arrondir vers zéro et mettre les sous-normaux à zéro. Vide si rien n'est à
// déclarer : ni rte ni denorm, module sans float 16 bits, ou module illisible.
std::vector<std::uint32_t> spirv_with_float16_controls(std::span<const std::uint32_t> words, bool rte,
                                                       bool denorm);

}  // namespace ajp::gpu::vk
