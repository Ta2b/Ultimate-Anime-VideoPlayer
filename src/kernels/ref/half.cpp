#include "kernels/ref/half.h"

#include <bit>

namespace ajp::kernels::ref {
namespace {

constexpr std::uint32_t kSignMask = 0x8000'0000U;
constexpr std::uint32_t kInfinity32 = 0x7F80'0000U;
constexpr std::uint32_t kFirstOverflow32 = 0x477F'F000U;   // 65520 : s'arrondit à l'infini
constexpr std::uint32_t kSmallestNormal32 = 0x3880'0000U;  // 2^−14, plus petit normal fp16
constexpr std::uint32_t kInfinity16 = 0x7C00U;
constexpr std::uint32_t kQuietNan16 = 0x7E00U;

// q + 1 si le reste dépasse la moitié, ou l'égale avec q impair.
constexpr std::uint32_t round_even(std::uint32_t q, std::uint32_t rest, std::uint32_t half) {
  return rest > half || (rest == half && (q & 1U) != 0) ? q + 1 : q;
}

}  // namespace

std::uint16_t float_to_half(float value) {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t sign = (bits & kSignMask) >> 16;
  const std::uint32_t magnitude = bits & ~kSignMask;
  std::uint32_t half = 0;
  if (magnitude > kInfinity32) {
    half = kQuietNan16;
  } else if (magnitude >= kFirstOverflow32) {
    half = kInfinity16;
  } else if (magnitude >= kSmallestNormal32) {
    // Exposant rebiaisé de 127 à 15, mantisse de 23 à 10 bits ; une retenue passe dans l'exposant.
    const std::uint32_t q = (((magnitude >> 23) - 112U) << 10) | ((magnitude >> 13) & 0x3FFU);
    half = round_even(q, magnitude & 0x1FFFU, 0x1000U);
  } else {
    // Sous-normal fp16 : unités de 2^−24, soit la mantisse complète décalée de 126 − exposant.
    const std::uint32_t exponent = magnitude >> 23;
    const std::uint32_t shift = 126U - exponent;
    if (shift < 25U) {
      const std::uint32_t mantissa = (magnitude & 0x7F'FFFFU) | 0x80'0000U;
      half = round_even(mantissa >> shift, mantissa & ((1U << shift) - 1U), 1U << (shift - 1U));
    }
  }
  return static_cast<std::uint16_t>(sign | half);
}

float half_to_float(std::uint16_t bits) {
  const std::uint32_t sign = (std::uint32_t{bits} & 0x8000U) << 16;
  const std::uint32_t exponent = (bits >> 10) & 0x1FU;
  const std::uint32_t mantissa = bits & 0x3FFU;
  if (exponent == 0x1FU) return std::bit_cast<float>(sign | kInfinity32 | (mantissa << 13));
  if (exponent != 0) return std::bit_cast<float>(sign | ((exponent + 112U) << 23) | (mantissa << 13));
  // Zéro ou sous-normal : mantisse × 2^−24, exact en float.
  const float magnitude = static_cast<float>(mantissa) * 0x1p-24F;
  return sign != 0 ? -magnitude : magnitude;
}

float round_to_half(float value) {
  return half_to_float(float_to_half(value));
}

RgbImage round_to_half(const RgbImage& image) {
  RgbImage out = image;
  for (PlaneF& p : out.planes) {
    for (float& v : p.data) v = round_to_half(v);
  }
  return out;
}

}  // namespace ajp::kernels::ref
