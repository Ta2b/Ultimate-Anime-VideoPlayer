// Les interpolateurs du système, que la chaîne charge comme un modèle RIFE v2 : un ONNX écrit par
// le lecteur (prepare/onnx_blend.h) porte cette marque, et le runtime qui la reconnaît (Core ML :
// VTFrameRateConversion de VideoToolbox, macOS 15.4) calcule l'image à l'instant t à la place du
// graphe. Une autre version de la marque est refusée, jamais prise pour le fondu.
#pragma once

#include <string_view>

namespace ajp::gpu {

inline constexpr std::string_view kSystemInterpolatorProducer = "ajp-system-interpolator";
inline constexpr std::string_view kVideoToolboxFrameRate = "videotoolbox-frame-rate-conversion-1";

}  // namespace ajp::gpu
