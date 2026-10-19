// Refus du socle CUDA : chaque code d'erreur CUDA devient un Refusal nommé qui cite l'appel.
#pragma once

#include <cuda_runtime_api.h>
#include <expected>
#include <string>
#include <string_view>

#include "core/types.h"

namespace ajp::gpu::cuda {

template <class T = void>
using Result = std::expected<T, Refusal>;

// cudaErrorMemoryAllocation donne GpuMemoryExhausted, tout autre code CudaFailed. Le détail
// cite l'appel, le nom de l'erreur et sa description, par exemple
// « cudaMalloc : cudaErrorMemoryAllocation (out of memory) ». L'erreur est aussi effacée de
// cudaGetLastError(), pour qu'un lancement suivant ne la reprenne pas à son compte.
Refusal cuda_refusal(cudaError_t error, std::string_view call);

// Rien si error vaut cudaSuccess, le refus de cuda_refusal sinon.
Result<> cuda_check(cudaError_t error, std::string_view call);

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail);

}  // namespace ajp::gpu::cuda
