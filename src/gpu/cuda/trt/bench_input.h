// Entrées du banc ajp_trt_bench, écrites une fois avant la mesure.
#pragma once

#include <cstddef>
#include <expected>
#include <string>
#include <vector>

#include "gpu/cuda/trt/types.h"

namespace ajp::gpu::trt::bench {

// Motif déterministe dans [0,5 ; 1[ : exposant fixe, mantisse variable. Le
// temps d'un réseau de convolutions comme SPAN n'en dépend pas.
std::vector<std::byte> noise_input(DataType type, std::size_t bytes);

// Entrée v1 de RIFE comme le lecteur la remplit (kernels::ref::rife_input) :
// deux images lisses, la seconde décalée de 3 px, instant 0,5, grille et pas.
// Le temps de RIFE dépend des données : une grille aléatoire, comme celle de
// trtexec, disperse les lectures de GridSample. F16, 1x11xHxW, H et W multiples de 64.
std::expected<std::vector<std::byte>, std::string> rife_input(DataType type, const Dims& dims);

}  // namespace ajp::gpu::trt::bench
