// INT8 symétrique des Q/DQ d'ONNX (QuantizeLinear à zéro INT8 nul) : q = arrondi au plus proche, pair en cas
// d'égalité, de x / échelle, saturé à [−128, 127] ; la valeur rendue est q × échelle.
#pragma once

#include <algorithm>
#include <cmath>

namespace ajp::gpu::ir {

inline constexpr float kInt8Max = 127.0F;
inline constexpr float kInt8Min = -128.0F;

// q de x (déjà divisé par l'échelle), en float exact ; NaN rend 0 (comme la saturation de TensorRT).
inline float round_to_int8(float value) {
  if (std::isnan(value)) return 0.0F;
  return std::clamp(std::nearbyint(value), kInt8Min, kInt8Max);  // pair en cas d'égalité (mode par défaut)
}

}  // namespace ajp::gpu::ir
