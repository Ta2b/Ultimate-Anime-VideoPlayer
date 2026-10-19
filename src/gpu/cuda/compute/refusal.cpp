#include "gpu/cuda/compute/refusal.h"

#include <format>
#include <utility>

namespace ajp::gpu::cuda {

Refusal cuda_refusal(cudaError_t error, std::string_view call) {
  // Efface l'erreur si elle n'est pas collante ; une erreur collante reste, et chaque appel
  // suivant la rendra.
  static_cast<void>(cudaGetLastError());
  const RefusalCode code =
      error == cudaErrorMemoryAllocation ? RefusalCode::GpuMemoryExhausted : RefusalCode::CudaFailed;
  return {.code = code,
          .detail = std::format("{} : {} ({})", call, cudaGetErrorName(error), cudaGetErrorString(error))};
}

Result<> cuda_check(cudaError_t error, std::string_view call) {
  if (error == cudaSuccess) return {};
  return std::unexpected(cuda_refusal(error, call));
}

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{.code = code, .detail = std::move(detail)});
}

}  // namespace ajp::gpu::cuda
