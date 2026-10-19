// État partagé entre un TrtRuntime et les moteurs qu'il a chargés, interne au
// module : TensorRT exige que le journal et l'IRuntime survivent à chaque
// ICudaEngine qui en est issu.
#pragma once

#include <NvInferRuntime.h>
#include <memory>
#include <utility>

#include "gpu/cuda/trt/trt_log.h"

namespace ajp::gpu::trt {

struct RuntimeState {
  explicit RuntimeState(std::shared_ptr<spdlog::logger> logger) : log(std::move(logger)) {}

  TrtLog log;                                   // déclaré d'abord : détruit en dernier
  std::unique_ptr<nvinfer1::IRuntime> runtime;  // créé avec log
};

}  // namespace ajp::gpu::trt
