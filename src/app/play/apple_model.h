// L'interpolation d'Apple (MotionEngine::Apple) se charge comme un modèle RIFE v2 : un ONNX écrit
// par le lecteur (prepare/onnx_blend.h) qui porte la marque de VTFrameRateConversion
// (gpu/system_interpolator.h), et que le runtime Core ML remplace par VideoToolbox. Écrit une fois
// dans le dossier de données du lecteur, comme les modèles dérivés (derived_model.h).
#pragma once

#include <expected>
#include <filesystem>

#include "core/types.h"

namespace ajp::app::play {

// dir/videotoolbox-frame-rate-conversion-1.onnx, écrit s'il manque ou diffère (écrit à côté puis
// renommé). Refus FileUnwritable.
std::expected<std::filesystem::path, Refusal> apple_model(const std::filesystem::path& dir);

}  // namespace ajp::app::play
