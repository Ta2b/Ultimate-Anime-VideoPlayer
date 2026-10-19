// Points d'entrée de ajp_backend_tensorrt.so (gpu/backend.h). Aucune exception ne traverse la
// frontière C : un échec devient nullptr et un refus dans init->refusal.
#include <dlfcn.h>
#include <filesystem>
#include <format>
#include <system_error>

#include "core/thread.h"
#include "gpu/backend.h"
#include "gpu/cuda/backend/tensorrt_backend.h"

namespace {

// trtexec livré à côté de la bibliothèque ; à défaut, celui trouvé à la configuration
// (AJP_TRTEXEC_PATH, poste de développement).
std::filesystem::path trtexec_path() {
  Dl_info info{};
  if (dladdr(reinterpret_cast<void*>(&ajp_backend_abi), &info) != 0 && info.dli_fname != nullptr) {
    const auto beside = std::filesystem::path(info.dli_fname).parent_path() / "trtexec";
    std::error_code error;
    if (std::filesystem::is_regular_file(beside, error)) return beside;
  }
  return AJP_TRTEXEC_PATH;
}

void tell(const ajp::gpu::BackendInit* init, ajp::Refusal refusal) {
  if (init != nullptr && init->refusal != nullptr) *init->refusal = std::move(refusal);
}

}  // namespace

extern "C" __attribute__((visibility("default"))) int ajp_backend_abi() noexcept {
  return ajp::gpu::kBackendAbi;
}

extern "C" __attribute__((visibility("default"))) std::uint64_t ajp_backend_layout() noexcept {
  return ajp::gpu::kBackendLayout;
}

extern "C" __attribute__((visibility("default"))) ajp::gpu::Backend* ajp_backend_create(
    int abi, const ajp::gpu::BackendInit* init) noexcept {
  if (init == nullptr) return nullptr;
  if (abi != ajp::gpu::kBackendAbi) {
    tell(init, {ajp::RefusalCode::AbiMismatch,
                std::format("backend tensorrt : ABI {} demandée, {} fournie", abi, ajp::gpu::kBackendAbi)});
    return nullptr;
  }
  try {
    auto backend = ajp::gpu::cuda::TensorRtBackend::create(*init, trtexec_path());
    if (!backend) {
      tell(init, std::move(backend.error()));
      return nullptr;
    }
    return backend->release();
  } catch (...) {
    tell(init, ajp::refusal_from_exception(ajp::RefusalCode::LibraryUnavailable, "backend tensorrt"));
    return nullptr;
  }
}

extern "C" __attribute__((visibility("default"))) void ajp_backend_destroy(
    ajp::gpu::Backend* backend) noexcept {
  delete backend;
}
