#include "app/play/backend_library.h"

#include <format>
#include <system_error>
#include <type_traits>
#include <utility>

namespace ajp::app::play {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

// Le point d'entrée name, ou nullptr. F est un pointeur de fonction (gpu/backend.h).
template <class F>
F symbol(const vendor_hdr::DynamicLibrary& library, const char* name) {
  auto found = library.symbol<std::remove_pointer_t<F>>(name);
  return found ? *found : nullptr;
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
    // dlopen sous Linux et macOS, LoadLibraryExW sous Windows (dépendances cherchées à côté).
    auto opened = vendor_hdr::DynamicLibrary::open(std::filesystem::absolute(path, error));
    if (!opened) {
      tried += std::format("\n  {}", opened.error().detail);
      continue;
    }
    library->handle_.emplace(std::move(*opened));
    library->path_ = path;
    const vendor_hdr::DynamicLibrary& handle = *library->handle_;
    const auto abi = symbol<gpu::BackendAbiFn>(handle, "ajp_backend_abi");
    const auto layout = symbol<gpu::BackendLayoutFn>(handle, "ajp_backend_layout");
    library->create_ = symbol<gpu::BackendCreateFn>(handle, "ajp_backend_create");
    library->destroy_ = symbol<gpu::BackendDestroyFn>(handle, "ajp_backend_destroy");
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

BackendLibrary::~BackendLibrary() = default;

void BackendLibrary::Destroy::operator()(gpu::Backend* backend) const {
  if (backend != nullptr) destroy(backend);
}

std::expected<std::shared_ptr<BackendLibrary>, Refusal> BackendLibrary::choose(
    std::span<const std::filesystem::path> candidates, const gpu::BackendInit& init) {
  std::string tried;
  for (const auto& path : candidates) {
    auto library = open(std::span(&path, 1));
    if (!library) {
      tried += std::format("\n  {}", library.error().detail);
      continue;
    }
    if (auto probe = create(*library, init); !probe) {
      tried += std::format("\n  {} : {} ({})", path.string(), probe.error().detail, name(probe.error().code));
      continue;
    }
    return library;
  }
  return refuse(RefusalCode::LibraryUnavailable, std::format("aucun backend pour cet appareil :{}", tried));
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
