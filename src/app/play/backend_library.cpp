#include "app/play/backend_library.h"

#include <dlfcn.h>
#include <format>
#include <system_error>
#include <utility>

namespace ajp::app::play {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

template <class F>
F symbol(void* handle, const char* name) {
  return reinterpret_cast<F>(dlsym(handle, name));
}

}  // namespace

std::expected<std::shared_ptr<BackendLibrary>, Refusal> BackendLibrary::open(
    std::span<const std::filesystem::path> candidates) {
  std::string tried;
  for (const auto& path : candidates) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error)) {
      tried += std::format("\n  {} : absent", path.string());
      continue;
    }
    std::shared_ptr<BackendLibrary> library(new BackendLibrary());
    library->handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (library->handle_ == nullptr) {
      tried += std::format("\n  {} : {}", path.string(), dlerror());
      continue;
    }
    library->path_ = path;
    const auto abi = symbol<gpu::BackendAbiFn>(library->handle_, "ajp_backend_abi");
    const auto layout = symbol<gpu::BackendLayoutFn>(library->handle_, "ajp_backend_layout");
    library->create_ = symbol<gpu::BackendCreateFn>(library->handle_, "ajp_backend_create");
    library->destroy_ = symbol<gpu::BackendDestroyFn>(library->handle_, "ajp_backend_destroy");
    if (abi == nullptr || layout == nullptr || library->create_ == nullptr || library->destroy_ == nullptr) {
      return refuse(RefusalCode::LibraryUnavailable,
                    std::format("{} : point d'entrée absent", path.string()));
    }
    if (auto ok = gpu::check_backend_abi(abi(), layout()); !ok) {
      return refuse(RefusalCode::AbiMismatch, std::format("{} : {}", path.string(), ok.error().detail));
    }
    return library;
  }
  return refuse(RefusalCode::LibraryUnavailable, std::format("aucune bibliothèque de backend :{}", tried));
}

BackendLibrary::~BackendLibrary() {
  if (handle_ != nullptr) dlclose(handle_);
}

void BackendLibrary::Destroy::operator()(gpu::Backend* backend) const {
  if (backend != nullptr) destroy(backend);
}

std::expected<BackendLibrary::Instance, Refusal> BackendLibrary::create(
    const std::shared_ptr<BackendLibrary>& library, const gpu::BackendInit& init) {
  Refusal refusal{RefusalCode::LibraryUnavailable, "ajp_backend_create sans refus"};
  gpu::BackendInit with_refusal = init;
  with_refusal.refusal = &refusal;
  gpu::Backend* backend = library->create_(gpu::kBackendAbi, &with_refusal);
  if (backend == nullptr) return std::unexpected(std::move(refusal));
  return Instance(backend, Destroy{.destroy = library->destroy_, .library = library});
}

}  // namespace ajp::app::play
