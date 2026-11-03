// Codes d'erreur CUDA en refus nommés qui citent l'appel.
#pragma once

#include <cuda_runtime_api.h>
#include <expected>
#include <format>
#include <string_view>

#include "core/types.h"

namespace ajp::gpu::cuda {

// Les erreurs que la documentation de CUDA décrit comme laissant le processus dans un état incohérent :
// le contexte est perdu, toute suite d'appels CUDA échoue jusqu'à la fin du processus.
constexpr bool sticky(cudaError_t error) {
  switch (error) {
    case cudaErrorIllegalAddress:
    case cudaErrorLaunchTimeout:
    case cudaErrorIllegalInstruction:
    case cudaErrorMisalignedAddress:
    case cudaErrorInvalidAddressSpace:
    case cudaErrorInvalidPc:
    case cudaErrorLaunchFailure:
    case cudaErrorHardwareStackError:
    case cudaErrorAssert:
    case cudaErrorECCUncorrectable:
    case cudaErrorContextIsDestroyed:
      return true;
    default:
      return false;
  }
}

inline Refusal cuda_refusal(cudaError_t error, std::string_view call) {
  if (sticky(error))
    return Refusal{RefusalCode::DeviceLost, std::format("contexte CUDA perdu : relancer le lecteur ({} : {})",
                                                        call, cudaGetErrorName(error))};
  return Refusal{RefusalCode::VendorCallFailed,
                 std::format("{} : {} ({})", call, cudaGetErrorName(error), cudaGetErrorString(error))};
}

// Lit et efface l'erreur en attente, sauf une erreur collante, qui ne s'efface pas : elle est rendue
// (cudaSuccess sinon).
inline cudaError_t clear_error() {
  const cudaError_t error = cudaGetLastError();
  return sticky(error) ? error : cudaSuccess;
}

inline std::expected<void, Refusal> cuda_check(cudaError_t error, std::string_view call) {
  if (error == cudaSuccess) return {};
  return std::unexpected(cuda_refusal(error, call));
}

}  // namespace ajp::gpu::cuda
