// Les appels CUDA dont l'échec n'arrête rien (libération, destruction, mesure, versions) : journalisés
// au canal « tensorrt » et l'erreur en attente effacée (P2.27), au lieu d'être ignorés en silence. Une
// erreur collante reste : le prochain appel vérifié la rend en DeviceLost. Pour les fichiers .cpp
// seulement (core/log.h n'entre pas dans nvcc).
#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <string_view>

#include "core/log.h"
#include "gpu/cuda/backend/cuda_status.h"

namespace ajp::gpu::cuda {

inline void cuda_warn(cudaError_t error, std::string_view call) noexcept {
  if (error == cudaSuccess) return;
  static_cast<void>(clear_error());
  try {
    log_channel("tensorrt")->warn("{} : {} ({})", call, cudaGetErrorName(error), cudaGetErrorString(error));
  } catch (...) {  // NOLINT(bugprone-empty-catch) : le journal même a échoué, rien d'autre à faire
  }
}

inline void cuda_warn(CUresult result, std::string_view call) noexcept {
  if (result == CUDA_SUCCESS) return;
  const char* label = nullptr;
  if (cuGetErrorName(result, &label) != CUDA_SUCCESS || label == nullptr) label = "CUresult inconnu";
  try {
    log_channel("tensorrt")->warn("{} : {}", call, label);
  } catch (...) {  // NOLINT(bugprone-empty-catch) : le journal même a échoué, rien d'autre à faire
  }
}

}  // namespace ajp::gpu::cuda
