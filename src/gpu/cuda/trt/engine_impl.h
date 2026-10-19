// Intérieur de TrtEngine, partagé par engine.cpp, engine_io.cpp et runtime.cpp.
#pragma once

#include <NvInferRuntime.h>
#include <expected>
#include <memory>
#include <vector>

#include "core/types.h"
#include "gpu/cuda/trt/engine.h"
#include "gpu/cuda/trt/runtime_state.h"
#include "gpu/cuda/trt/types.h"

namespace ajp::gpu::trt {

// Les membres se détruisent dans l'ordre inverse : contexte, moteur, puis
// l'état du runtime qui doit leur survivre.
struct TrtEngine::Impl {
  std::shared_ptr<RuntimeState> state;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  std::vector<TensorDesc> tensors;
  std::vector<void*> addresses;  // une par tenseur, nullptr tant que bind() ne l'a pas posée
  int profile = 0;
};

// Décrit les entrées et sorties du moteur et leurs profils. Refuse ce que le
// runtime ne sait pas lier : tenseur de forme lu sur l'hôte, format vectorisé,
// type sans taille connue.
std::expected<std::vector<TensorDesc>, Refusal> describe_io(const nvinfer1::ICudaEngine& engine);

}  // namespace ajp::gpu::trt
