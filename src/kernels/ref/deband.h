// Référence du débandage de la source avant SPAN (KernelOps::deband) : l'algorithme de libplacebo
// (pl_shader_deband, une itération, sans grain), tel que le font le noyau CUDA
// (gpu/cuda/backend/deband_kernels.cuh) et le shader Vulkan (kernels/glsl/deband.comp). Ici plutôt que dans
// spec/ : le débandage ne passe pas la frontière des moteurs (KernelOps::deband ne prend qu'un tenseur), et
// un en-tête de spec/ en plus changerait l'empreinte de la frontière.
//
// Un port lit les mêmes quatre points en croix et suit la même règle ; seul le tirage du décalage passe par
// sin et cos, que GLSL ne fixe pas au bit près : un décalage peut tomber à un pixel près sur une demi-unité,
// et le pixel prend alors la moyenne d'autres points, à moins de deux seuils de la référence (kDebandSpread).
// Tests : tests/unit/kernels/deband_test.cpp, tests/unit/vk_backend/deband_test.cpp.
#pragma once

#include <cstdint>

#include "kernels/image.h"

namespace ajp::kernels::ref {

// Rayon et seuil de libplacebo (pl_deband_default_params : 16 pixels, 3 / 1000 du code), les valeurs de
// gpu/cuda/backend/deband_kernels.h : un pixel prend la moyenne de quatre points tirés dans son rayon tant
// qu'elle reste sous le seuil, canal par canal ; un vrai trait s'en écarte davantage et reste net.
inline constexpr float kDebandRadius = 16.0F;
inline constexpr float kDebandThreshold = 3.0F / 1000.0F;
// L'écart d'un port à la référence quand son décalage diffère : les deux valeurs restent à moins d'un seuil
// de la source.
inline constexpr float kDebandSpread = 2.0F * kDebandThreshold;

// Bruit uniforme dans [0, 1) tiré d'un hachage entier (PCG) du pixel : fixe d'une image à l'autre, pour
// qu'un aplat immobile ne scintille pas.
constexpr float deband_uniform(std::uint32_t x, std::uint32_t y, std::uint32_t salt) {
  std::uint32_t v = (x * 0x9E3779B1U) ^ (y * 0x85EBCA77U) ^ (salt * 0xC2B2AE3DU);
  v = (v * 747796405U) + 2891336453U;
  v = ((v >> ((v >> 28U) + 4U)) ^ v) * 277803737U;
  v = (v >> 22U) ^ v;
  return static_cast<float>(v >> 8U) * (1.0F / 16777216.0F);
}

struct DebandOffset {
  int dx = 0;
  int dy = 0;
};

// Le décalage du pixel (x, y) : les quatre points sont (±dx, ±dy) et (∓dy, ±dx), arrondis au plus proche
// (moitié vers le pair, comme __float2int_rn et roundEven).
DebandOffset deband_offset(int x, int y);

// La règle : la moyenne des quatre points, sauf si elle s'écarte de la valeur de plus du seuil.
float deband_rule(float value, float a, float b, float d, float e);

// L'image débandée, de la taille de la source ; les points hors de l'image sont ramenés au bord.
Result<RgbImage> deband(const RgbImage& image);

}  // namespace ajp::kernels::ref
