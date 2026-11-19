#include "gpu/ir/fp8_weights.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "gpu/ir/fp8.h"
#include "gpu/ir/int8.h"

namespace kirari::gpu::ir {
namespace {

// Les valeurs de [first, last) tombent-elles sur la grille E4M3 de cette échelle ? À un arrondi
// float16 près (2^-10 de la valeur : un modèle float16 range q × échelle en demi-flottants) ; une
// mauvaise échelle laisse des écarts d'un demi-pas de la grille, de l'ordre de 2^-4 de la valeur. Sous 2^-14,
// un float16 est dénormal, au pas absolu de 2^-24 : son arrondi en écarte jusqu'à 2^-25 (un canal de petits
// poids, sous 2^-4, du paquet du SPAN FP8, gpu/span_fp8/package_model.h).
// La grille INT8 est plus grossière (un pas de 1/127 de amax au moins) : une mauvaise échelle s'y voit
// autant.
bool on_grid(const std::vector<float>& values, std::size_t first, std::size_t last, float scale, float amax,
             QuantGrid grid = QuantGrid::E4m3) {
  constexpr float kHalfSubnormalError = 0x1p-25F;
  for (std::size_t k = first; k < last; ++k) {
    const float rebuilt = quant_round(grid, values[k], scale);
    const float tolerance =
        std::max({std::ldexp(std::fabs(values[k]), -10), std::ldexp(amax, -21), kHalfSubnormalError});
    if (!(std::fabs(rebuilt - values[k]) <= tolerance)) return false;
  }
  return true;
}

float max_abs(const std::vector<float>& values, std::size_t first, std::size_t last) {
  float amax = 0.0F;
  for (std::size_t k = first; k < last; ++k) amax = std::max(amax, std::fabs(values[k]));
  return amax;
}

// Les poids d'une Conv de rang 4, finis : le nombre de canaux de sortie, sinon 0.
std::size_t channels_of(const Value& weights) {
  if (weights.type == DataType::I64 || weights.dims.size() != 4 || weights.f.empty()) return 0;
  const auto cout = static_cast<std::size_t>(weights.dims[0]);
  if (cout == 0 || weights.f.size() % cout != 0) return 0;
  if (std::ranges::any_of(weights.f, [](float v) { return !std::isfinite(v); })) return 0;
  return cout;
}

// L'échelle INT8 d'un canal : amax / k pour le premier k (de 128 à 1) qui range tout le canal sur la grille.
// 128 d'abord : un canal dont le plus grand en valeur absolue vaut -128 × échelle (l'arrondi saturant à 127
// écarte un +128, on_grid le refuse).
std::optional<float> int8_scale(const std::vector<float>& values, std::size_t first, std::size_t last) {
  const float amax = max_abs(values, first, last);
  if (!(amax > 0.0F)) return 1.0F;
  for (int k = 128; k >= 1; --k) {
    const float scale = amax / static_cast<float>(k);
    if (on_grid(values, first, last, scale, amax, QuantGrid::Int8)) return scale;
  }
  return std::nullopt;
}

}  // namespace

std::optional<Int8Weights> recover_int8_weights(const Value& weights) {
  const std::size_t cout = channels_of(weights);
  if (cout == 0) return std::nullopt;
  const std::size_t per_channel = weights.f.size() / cout;
  Int8Weights out;
  out.scale.assign(cout, 1.0F);
  out.bytes.resize(weights.f.size());
  for (std::size_t c = 0; c < cout; ++c) {
    const std::size_t first = c * per_channel;
    const std::size_t last = first + per_channel;
    const std::optional<float> scale = int8_scale(weights.f, first, last);
    if (!scale) return std::nullopt;
    out.scale[c] = *scale;
    for (std::size_t k = first; k < last; ++k) {
      const auto q = static_cast<std::int32_t>(round_to_int8(weights.f[k] / *scale));
      out.bytes[k] = static_cast<std::uint8_t>(q & 0xFF);
    }
  }
  return out;
}

std::optional<Fp8Weights> recover_fp8_weights(const Value& weights) {
  const std::size_t cout = channels_of(weights);
  if (cout == 0) return std::nullopt;
  const std::size_t per_channel = weights.f.size() / cout;

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

}  // namespace kirari::gpu::ir
