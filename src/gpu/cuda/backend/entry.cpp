// Points d'entrée de ajp_backend_tensorrt.so (gpu/backend.h). Aucune exception ne traverse la
// frontière C : un échec devient nullptr et un refus dans init->refusal.
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <filesystem>
#include <format>
#include <string>
#include <system_error>

#include "core/thread.h"
#include "gpu/backend.h"
#include "gpu/cuda/backend/tensorrt_backend.h"

namespace {

// Le fichier de cette bibliothèque ; vide s'il échappe.
std::filesystem::path own_file() {
#ifdef _WIN32
  HMODULE self = nullptr;
  if (GetModuleHandleExW(
          GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          reinterpret_cast<LPCWSTR>(&own_file), &self) == 0)
    return {};
  std::wstring name(MAX_PATH, L'\0');
  for (;;) {
    const DWORD length = GetModuleFileNameW(self, name.data(), static_cast<DWORD>(name.size()));
    if (length == 0) return {};
    if (length < name.size()) return std::filesystem::path(name.substr(0, length));
    name.resize(name.size() * 2);
  }
#else
  Dl_info info{};
  if (dladdr(reinterpret_cast<void*>(&own_file), &info) == 0 || info.dli_fname == nullptr) return {};
  return info.dli_fname;
#endif
}

// trtexec livré à côté de la bibliothèque ; à défaut, celui trouvé à la configuration
// (AJP_TRTEXEC_PATH, poste de développement).
std::filesystem::path trtexec_path() {
#ifdef _WIN32
  constexpr const char* kTrtexec = "trtexec.exe";
#else
  constexpr const char* kTrtexec = "trtexec";
#endif
  if (const std::filesystem::path self = own_file(); !self.empty()) {
    const auto beside = self.parent_path() / kTrtexec;
    std::error_code error;
    if (std::filesystem::is_regular_file(beside, error)) return beside;
  }
  return AJP_TRTEXEC_PATH;
}

void tell(const ajp::gpu::BackendInit* init, ajp::Refusal refusal) {
  if (init != nullptr && init->refusal != nullptr) *init->refusal = std::move(refusal);
}

}  // namespace

extern "C" AJP_BACKEND_EXPORT int ajp_backend_abi() noexcept {
  return ajp::gpu::kBackendAbi;
}

extern "C" AJP_BACKEND_EXPORT std::uint64_t ajp_backend_layout() noexcept {
  return ajp::gpu::kBackendLayout;
}

extern "C" AJP_BACKEND_EXPORT ajp::gpu::Backend* ajp_backend_create(
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

extern "C" AJP_BACKEND_EXPORT void ajp_backend_destroy(ajp::gpu::Backend* backend) noexcept {
  delete backend;
}
