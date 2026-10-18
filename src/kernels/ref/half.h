// Arrondi fp16 des tenseurs des réseaux : les ports écrivent leurs plans RGB en binary16 par
// __float2half (CUDA), packHalf2x16 (GLSL) ou half (Metal), au plus proche, égalités au pair.
// La référence calcule en float ; ces fonctions reproduisent l'arrondi d'un port pour qu'un
// golden compare des valeurs comparables. Tests : tests/unit/kernels/half_test.cpp.
#pragma once

#include <cstdint>

#include "kernels/image.h"

namespace ajp::kernels::ref {

// float → bits binary16 : au plus proche, égalités au pair ; ≥ 65520 en valeur absolue donne
// l'infini, NaN reste NaN (silencieux), les sous-normaux sont conservés.
std::uint16_t float_to_half(float value);
float half_to_float(std::uint16_t bits);

// La valeur qu'un port relit après l'avoir rangée en fp16.
float round_to_half(float value);

// Chaque échantillon de l'image arrondi en fp16.
RgbImage round_to_half(const RgbImage& image);

}  // namespace ajp::kernels::ref
