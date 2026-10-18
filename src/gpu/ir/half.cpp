#include "gpu/ir/half.h"

#include <bit>

namespace ajp::gpu::ir {

float half_to_float(std::uint16_t bits) {
  const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16U;
  const std::uint32_t exponent = (bits >> 10U) & 0x1FU;
  std::uint32_t mantissa = bits & 0x3FFU;

  if (exponent == 0x1FU) return std::bit_cast<float>(sign | 0x7F800000U | (mantissa << 13U));
  if (exponent != 0) return std::bit_cast<float>(sign | ((exponent + 112U) << 23U) | (mantissa << 13U));
  if (mantissa == 0) return std::bit_cast<float>(sign);

  // Sous-normal : on normalise la mantisse.
  std::uint32_t shifted_exponent = 113;
  while ((mantissa & 0x400U) == 0) {
    mantissa <<= 1U;
    --shifted_exponent;
  }
  return std::bit_cast<float>(sign | (shifted_exponent << 23U) | ((mantissa & 0x3FFU) << 13U));
}

std::uint16_t float_to_half(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const auto sign = static_cast<std::uint16_t>((bits >> 16U) & 0x8000U);
  const std::uint32_t exponent = (bits >> 23U) & 0xFFU;
  const std::uint32_t mantissa = bits & 0x7FFFFFU;

  if (exponent == 0xFFU) {
    // Infini, ou NaN dont on garde un bit de mantisse pour qu'il reste NaN.
    const std::uint32_t nan = mantissa != 0 ? 0x200U | (mantissa >> 13U) : 0U;
    return static_cast<std::uint16_t>(sign | 0x7C00U | nan);
  }

  const int unbiased = static_cast<int>(exponent) - 127;
  if (unbiased > 15) return static_cast<std::uint16_t>(sign | 0x7C00U);

  // Mantisse complète avec le bit implicite, puis décalage vers binary16.
  const std::uint32_t full = mantissa | 0x800000U;
  int shift = 13;
  std::uint32_t half_exponent = 0;
  if (unbiased >= -14) {
    half_exponent = static_cast<std::uint32_t>(unbiased + 15);
  } else {
    shift = 13 + (-14 - unbiased);  // sous-normal
    if (shift > 24) return sign;    // trop petit : zéro signé
  }

  const auto ushift = static_cast<std::uint32_t>(shift);
  std::uint32_t result = full >> ushift;
  const std::uint32_t remainder = full & ((1U << ushift) - 1U);
  const std::uint32_t halfway = 1U << (ushift - 1U);
  if (remainder > halfway || (remainder == halfway && (result & 1U) != 0)) ++result;

  if (half_exponent == 0) {
    // Sous-normal : l'arrondi peut remonter jusqu'au plus petit normal (0x400).
    return static_cast<std::uint16_t>(sign | result);
  }
  // result contient le bit implicite (0x400) ; un report peut incrémenter l'exposant.
  const std::uint32_t combined = (half_exponent << 10U) + (result - 0x400U);
  if (combined >= 0x7C00U) return static_cast<std::uint16_t>(sign | 0x7C00U);
  return static_cast<std::uint16_t>(sign | combined);
}

float round_to_half(float value) {
  return half_to_float(float_to_half(value));
}

}  // namespace ajp::gpu::ir
