// Étape 10.3 : les échelles des poids FP8 (E4M3) des convolutions de bloc, par canal de sortie. Poids rangés
// [sortie][ligne][colonne][entrée] (package.h) : le canal de sortie c tient les éléments
// [c·per_channel, (c + 1)·per_channel). Échelle = maximum de |poids| du canal / 448 (1 pour un canal nul) ;
// le noyau multiplie l'accumulateur de la colonne c par (échelle de l'activation × échelle du canal).
#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace ajp::gpu::span_fp8 {

inline constexpr float kE4m3Max = 448.0F;

// x arrondi au plus proche E4M3 (pair en cas d'égalité), saturé à ±448 : la conversion
// __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3), sur l'hôte.
float e4m3_round(float x);

// Une échelle par canal de sortie (out_channels canaux de per_channel poids).
std::vector<float> channel_scales(std::span<const float> weights, std::size_t out_channels);
// Une seule échelle pour tout le tenseur (l'état d'avant l'étape 10.3).
float tensor_scale(std::span<const float> weights);

// Les poids tels que le noyau les voit : e4m3_round(w / échelle du canal) × échelle du canal.
std::vector<float> reconstruct(std::span<const float> weights, std::span<const float> scales);

}  // namespace ajp::gpu::span_fp8
