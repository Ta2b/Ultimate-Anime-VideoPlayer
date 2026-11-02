// Ce qu'un module SPIR-V promet de ses tampons de stockage : les bindings (jeu 0) que toutes ses variables
// déclarent readonly (décoration NonWritable sur la variable ou sur chaque membre de son bloc). Le Recorder
// ne pose une barrière entre deux dispatchs que si l'un écrit ce que l'autre touche : deux lectures d'un même
// tampon, ou du tampon factice d'un binding inutilisé, se recouvrent.
#pragma once

#include <cstdint>
#include <span>

namespace ajp::gpu::vk {

// Un bit par binding de 0 à 63 ; 0 pour un module illisible (tout compte alors comme écrit).
std::uint64_t spirv_readonly_bindings(std::span<const std::uint32_t> words);

}  // namespace ajp::gpu::vk
