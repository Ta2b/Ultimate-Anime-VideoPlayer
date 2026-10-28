// FP8 E4M3 (variante « fn » d'ONNX et de TensorRT) : 1 bit de signe, 4 d'exposant (biais 7), 3 de
// mantisse, pas d'infini ; 0x7F et 0xFF sont NaN, la plus grande valeur finie est 448.
// L'arrondi est au plus proche, pair en cas d'égalité, saturant : au-delà de 448, la valeur reste 448.
#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>

namespace ajp::gpu::ir {

inline constexpr float kE4m3Max = 448.0F;

inline float e4m3_to_float(std::uint8_t bits) {
  const std::uint32_t sign = bits >> 7U;
  const std::uint32_t exponent = (bits >> 3U) & 0xFU;
  const std::uint32_t mantissa = bits & 0x7U;
  float magnitude = 0.0F;
  if ((bits & 0x7FU) == 0x7FU) {
    magnitude = std::bit_cast<float>(0x7FC00000U);
  } else if (exponent == 0) {
    magnitude = std::ldexp(static_cast<float>(mantissa), -9);  // sous-normaux : m / 8 × 2^-6
  } else {
    magnitude = std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F, static_cast<int>(exponent) - 7);
  }
  return sign != 0 ? -magnitude : magnitude;
}

inline std::uint8_t float_to_e4m3(float value) {
  if (std::isnan(value)) return 0x7F;
  const std::uint8_t sign = std::signbit(value) ? 0x80U : 0U;
  float magnitude = std::fabs(value);
  if (magnitude >= kE4m3Max) return static_cast<std::uint8_t>(sign | 0x7EU);  // saturation à 448
  // Pas de la grille autour de magnitude : 2^-9 sous 2^-6, sinon 2^(e - 3).
  int exponent = 0;
  std::frexp(magnitude, &exponent);  // magnitude = f × 2^exponent, f dans [0,5 ; 1[
  const int step_exponent = std::max(exponent - 1, -6) - 3;
  const float step = std::ldexp(1.0F, step_exponent);
  const float steps = std::nearbyint(magnitude / step);  // pair en cas d'égalité (mode par défaut)
  magnitude = steps * step;
  if (magnitude >= kE4m3Max) return static_cast<std::uint8_t>(sign | 0x7EU);
  if (magnitude == 0.0F) return sign;
  int e = 0;
  const float f = std::frexp(magnitude, &e);  // [0,5 ; 1[
  std::uint32_t biased = 0;
  std::uint32_t mantissa = 0;
  if (e - 1 < -6) {
    mantissa = static_cast<std::uint32_t>(std::lround(std::ldexp(magnitude, 9)));
  } else {
    biased = static_cast<std::uint32_t>(e - 1 + 7);
    mantissa = static_cast<std::uint32_t>(std::lround((f * 2.0F - 1.0F) * 8.0F));
  }
  return static_cast<std::uint8_t>(sign | (biased << 3U) | mantissa);
}

inline float round_to_e4m3(float value) {
  return e4m3_to_float(float_to_e4m3(value));
}

}  // namespace ajp::gpu::ir
