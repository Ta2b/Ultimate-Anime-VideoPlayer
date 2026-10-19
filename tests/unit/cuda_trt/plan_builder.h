// Construction en processus d'un moteur TensorRT depuis un petit ONNX généré
// dans le test, et outils communs des tests [cuda]. Le lecteur, lui, construit
// toujours par trtexec en sous-processus (prepare).
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "core/types.h"
#include "gpu/cuda/trt/types.h"

namespace ajp::gpu::trt {
class TrtLog;
}

namespace ajp::gpu::trt::test {

// Vide si un appareil CUDA est utilisable ; sinon la raison, pour SKIP.
std::string cuda_unavailable_reason();

// Moteur sérialisé ; profile borne l'entrée nommée input quand elle est
// dynamique. FP32 sans TF32, pour comparer à l'interpréteur au plus près.
// log est celui du runtime vivant : TensorRT n'en garde qu'un par processus.
std::expected<std::vector<std::byte>, std::string> build_plan(TrtLog& log, std::span<const std::byte> onnx,
                                                              const std::string& input,
                                                              const std::optional<ShapeRange>& profile);

// Copie vers l'appareil, et retour.
std::expected<void, Refusal> upload(void* device, std::span<const std::byte> host);
std::expected<std::vector<std::byte>, Refusal> download(const void* device, std::size_t bytes);

// F32 et F16 petit-boutistes vers float, et inversement.
std::vector<std::byte> to_bytes(DataType type, std::span<const float> values);
std::vector<float> to_floats(DataType type, std::span<const std::byte> bytes);

}  // namespace ajp::gpu::trt::test
