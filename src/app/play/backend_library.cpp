#include "app/play/backend_library.h"

#include <format>
#include <mutex>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace ajp::app::play {
namespace {

// Le point d'entrée name, ou nullptr. F est un pointeur de fonction (gpu/backend.h).
template <class F>
F symbol(const DynamicLibrary& library, const char* name) {
  auto found = library.symbol<std::remove_pointer_t<F>>(name);
  return found ? *found : nullptr;
}

struct Kept {
  std::mutex mutex;
  std::vector<std::shared_ptr<BackendLibrary>> libraries;
};

Kept& kept() {
  static Kept instance;
  return instance;
}

// Les bibliothèques gardées se referment ici, hors du verrou, sauf celle que choose rend.
void release_kept() {
  std::vector<std::shared_ptr<BackendLibrary>> released;
  {
    Kept& k = kept();
    const std::scoped_lock lock(k.mutex);
    released.swap(k.libraries);
  }
}

}  // namespace

void BackendLibrary::keep_open(std::shared_ptr<BackendLibrary> library) {
  if (library == nullptr) return;
  Kept& k = kept();
  const std::scoped_lock lock(k.mutex);
  k.libraries.push_back(std::move(library));
}

std::expected<std::shared_ptr<BackendLibrary>, Refusal> BackendLibrary::open(
    std::span<const std::filesystem::path> candidates) {
  std::string tried;
  for (const auto& path : candidates) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error)) {
      tried += std::format("\n  {} : absent", path.string());
      continue;
    }
    {
      Kept& k = kept();
      const std::scoped_lock lock(k.mutex);
      for (const auto& held : k.libraries)
        if (held->path_ == path && !held->replaced_on_disk()) return held;  // même chemin que open l'a reçu
    }
    std::shared_ptr<BackendLibrary> library(new BackendLibrary());
    // dlopen sous Linux et macOS, LoadLibraryExW sous Windows (dépendances cherchées à côté).
    auto opened = DynamicLibrary::open(std::filesystem::absolute(path, error));
    if (!opened) {
      tried += std::format("\n  {}", opened.error().detail);
      continue;
    }
    library->handle_.emplace(std::move(*opened));
    library->path_ = path;
    library->stamp_ = std::filesystem::last_write_time(path, error);
    const DynamicLibrary& handle = *library->handle_;
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

bool BackendLibrary::replaced_on_disk() const {
  std::error_code error;
  const auto now = std::filesystem::last_write_time(path_, error);
  return !error && now != stamp_;
}

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
      if (init.log)
        init.log(gpu::BackendLogLevel::Warn, std::format("backend écarté : {}", library.error().detail));
      continue;
    }
    if (auto probe = create(*library, init); !probe) {
      const std::string why =
          std::format("{} : {} ({})", path.string(), probe.error().detail, name(probe.error().code));
      tried += "\n  " + why;
      if (init.log) init.log(gpu::BackendLogLevel::Warn, "backend écarté : " + why);
      continue;
    }
    (*library)->passed_over_ = std::move(tried);
    release_kept();
    return library;
  }
  release_kept();
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
