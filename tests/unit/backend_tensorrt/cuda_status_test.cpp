#include "gpu/cuda/backend/cuda_status.h"

#include <catch2/catch_test_macros.hpp>

namespace ajp::gpu::cuda {

TEST_CASE("CUDA : une erreur collante est un appareil perdu, les autres un appel refusé",
          "[backend_tensorrt][cuda]") {
  CHECK(sticky(cudaErrorIllegalAddress));
  CHECK(sticky(cudaErrorLaunchFailure));
  CHECK(sticky(cudaErrorLaunchTimeout));
  CHECK_FALSE(sticky(cudaSuccess));
  CHECK_FALSE(sticky(cudaErrorMemoryAllocation));
  CHECK_FALSE(sticky(cudaErrorInvalidValue));
  CHECK(cuda_refusal(cudaErrorIllegalAddress, "cudaGraphLaunch").code == RefusalCode::DeviceLost);
  CHECK(cuda_refusal(cudaErrorInvalidValue, "cudaGraphLaunch").code == RefusalCode::VendorCallFailed);
  CHECK(clear_error() == cudaSuccess);  // aucune erreur en attente dans ce processus
}

}  // namespace ajp::gpu::cuda
