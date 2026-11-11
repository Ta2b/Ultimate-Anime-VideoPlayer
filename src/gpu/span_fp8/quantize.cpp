#include "gpu/span_fp8/quantize.h"

#include <algorithm>
#include <cmath>

namespace ajp::gpu::span_fp8 {

float e4m3_round(float x) {
  if (std::isnan(x)) return x;
  const float a = std::fabs(x);
  if (a == 0.0F) return x;
  // E4M3 : 3 bits de mantisse, exposant minimal normal −6 ; sous 2^−6 le pas est celui des dénormaux, 2^−9.
  int exponent = 0;
  std::frexp(a, &exponent);  // a = m · 2^exponent, m dans [0,5, 1)
  const int e = std::max(exponent - 1, -6);
  const float step = std::ldexp(1.0F, e - 3);
  float q = std::nearbyint(a / step) * step;  // au plus proche, pair en cas d'égalité
  q = std::min(q, kE4m3Max);
  return std::copysign(q, x);
}

float tensor_scale(std::span<const float> weights) {
  float peak = 0.0F;
  for (const float w : weights) peak = std::max(peak, std::fabs(w));
  return peak > 0.0F ? peak / kE4m3Max : 1.0F;
}

std::vector<float> channel_scales(std::span<const float> weights, std::size_t out_channels) {
  std::vector<float> scales(out_channels, 1.0F);
  if (out_channels == 0) return scales;
  // Arrondi vers le haut : un reste va au dernier canal au lieu d'être ignoré (B02-11) ; le compte exact,
  // seul cas du lecteur, ne change pas.
  const std::size_t per = (weights.size() + out_channels - 1) / out_channels;
  for (std::size_t c = 0; c < out_channels; ++c) {
    const std::size_t first = std::min(c * per, weights.size());
    scales[c] = tensor_scale(weights.subspan(first, std::min(per, weights.size() - first)));
  }
  return scales;
}

std::vector<float> reconstruct(std::span<const float> weights, std::span<const float> scales) {
  std::vector<float> out(weights.size());
  if (scales.empty()) return out;
  // Même découpage que channel_scales ; jamais nul, même avec plus d'échelles que de poids (B02-11).
  const std::size_t per = std::max<std::size_t>(1, (weights.size() + scales.size() - 1) / scales.size());
  for (std::size_t i = 0; i < weights.size(); ++i) {
    const float s = scales[std::min(i / per, scales.size() - 1)];
    out[i] = e4m3_round(weights[i] / s) * s;
  }
  return out;
}

}  // namespace ajp::gpu::span_fp8
