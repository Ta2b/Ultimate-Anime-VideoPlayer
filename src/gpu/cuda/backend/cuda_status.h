// Codes d'erreur CUDA en refus nommés qui citent l'appel.
#pragma once

#include <cuda_runtime_api.h>
#include <expected>
#include <format>
#include <string_view>

#include "core/types.h"

namespace ajp::gpu::cuda {

inline Refusal cuda_refusal(cudaError_t error, std::string_view call) {
  return Refusal{RefusalCode::VendorCallFailed,
                 std::format("{} : {} ({})", call, cudaGetErrorName(error), cudaGetErrorString(error))};
}

inline std::expected<void, Refusal> cuda_check(cudaError_t error, std::string_view call) {
  if (error == cudaSuccess) return {};
  return std::unexpected(cuda_refusal(error, call));
}

}  // namespace ajp::gpu::cuda
