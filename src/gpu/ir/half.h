// Conversions IEEE 754 binary16 <-> binary32, sans dépendance matérielle.
#pragma once

#include <cstdint>

namespace ajp::gpu::ir {

// Exacte : tout binary16 (sous-normaux, infinis, NaN compris) tient dans un float.
float half_to_float(std::uint16_t bits);

// Arrondi au plus proche, égalité vers le pair ; au-delà de 65504, infini.
std::uint16_t float_to_half(float value);

// float_to_half puis half_to_float : la valeur F16 la plus proche.
float round_to_half(float value);

}  // namespace ajp::gpu::ir
