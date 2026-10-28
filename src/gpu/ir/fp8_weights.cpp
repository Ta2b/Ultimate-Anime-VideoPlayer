#include "gpu/ir/fp8_weights.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "gpu/ir/fp8.h"

namespace ajp::gpu::ir {
namespace {

// Les valeurs de [first, last) tombent-elles sur la grille E4M3 de cette échelle ? À un arrondi
// float16 près (2^-10 de la valeur : un modèle float16 range q × échelle en demi-flottants) ; une
// mauvaise échelle laisse des écarts d'un demi-pas de la grille, de l'ordre de 2^-4 de la valeur.
bool on_grid(const std::vector<float>& values, std::size_t first, std::size_t last, float scale, float amax) {
  for (std::size_t k = first; k < last; ++k) {
    const float rebuilt = round_to_e4m3(values[k] / scale) * scale;
    const float tolerance = std::max(std::ldexp(std::fabs(values[k]), -10), std::ldexp(amax, -21));
    if (!(std::fabs(rebuilt - values[k]) <= tolerance)) return false;
  }
  return true;
}

float max_abs(const std::vector<float>& values, std::size_t first, std::size_t last) {
  float amax = 0.0F;
  for (std::size_t k = first; k < last; ++k) amax = std::max(amax, std::fabs(values[k]));
  return amax;
}

}  // namespace

std::optional<Fp8Weights> recover_fp8_weights(const Value& weights) {
  if (weights.type == DataType::I64 || weights.dims.size() != 4 || weights.f.empty()) return std::nullopt;
  const auto cout = static_cast<std::size_t>(weights.dims[0]);
  if (cout == 0 || weights.f.size() % cout != 0) return std::nullopt;
  const std::size_t per_channel = weights.f.size() / cout;
  if (std::ranges::any_of(weights.f, [](float v) { return !std::isfinite(v); })) return std::nullopt;

  const float tensor_amax = max_abs(weights.f, 0, weights.f.size());
  const float tensor_scale = tensor_amax > 0.0F ? tensor_amax / kE4m3Max : 1.0F;
  Fp8Weights out;
  out.scale.assign(cout, tensor_scale);
  out.bytes.resize(weights.f.size());
  for (std::size_t c = 0; c < cout; ++c) {
    const std::size_t first = c * per_channel;
    const std::size_t last = first + per_channel;
    const float amax = max_abs(weights.f, first, last);
    float scale = tensor_scale;
    // Échelle du tenseur d'abord, puis celle du canal.
    if (!on_grid(weights.f, first, last, scale, amax)) {
      scale = amax > 0.0F ? amax / kE4m3Max : 1.0F;
      if (!on_grid(weights.f, first, last, scale, amax)) return std::nullopt;
    }
    out.scale[c] = scale;
    for (std::size_t k = first; k < last; ++k) out.bytes[k] = float_to_e4m3(weights.f[k] / scale);
  }
  return out;
}

}  // namespace ajp::gpu::ir
