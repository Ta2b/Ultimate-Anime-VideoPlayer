// Points d'entrée de la bibliothèque ajp_backend_cpu (gpu/backend.h, gpu/rife_input.h). Seuls
// symboles exportés : le reste de la bibliothèque est caché à l'édition des liens.
#include <cstdint>
#include <format>
#include <new>

#include "core/thread.h"
#include "gpu/backend.h"
#include "gpu/cpu/cpu_backend.h"
#include "gpu/rife_input.h"

extern "C" int ajp_backend_abi() noexcept {
  return ajp::gpu::kBackendAbi;
}

extern "C" std::uint64_t ajp_backend_layout() noexcept {
  return ajp::gpu::kBackendLayout;
}

extern "C" ajp::gpu::Backend* ajp_backend_create(int abi, const ajp::gpu::BackendInit* init) noexcept {
  if (init == nullptr) return nullptr;
  if (abi != ajp::gpu::kBackendAbi) {
    if (init->refusal != nullptr) {
      *init->refusal = {ajp::RefusalCode::AbiMismatch,
                        std::format("backend cpu : ABI {} demandée, {} fournie", abi, ajp::gpu::kBackendAbi)};
    }
    return nullptr;
  }
  try {
    return new ajp::gpu::cpu::CpuBackend(*init);
  } catch (...) {
    if (init->refusal != nullptr) {
      *init->refusal = ajp::refusal_from_exception(ajp::RefusalCode::LibraryUnavailable, "backend cpu");
    }
    return nullptr;
  }
}

// La signature est celle de gpu/backend.h.
extern "C" void ajp_backend_destroy(ajp::gpu::Backend* backend) noexcept {  // NOLINT(misc-const-correctness)
  delete backend;
}

extern "C" ajp::gpu::RifeInputOps* ajp_backend_rife_input(ajp::gpu::Backend* backend) noexcept {
  // backend vient de ajp_backend_create de cette bibliothèque.
  return backend == nullptr ? nullptr : static_cast<ajp::gpu::cpu::CpuBackend*>(backend)->rife_input();
}
