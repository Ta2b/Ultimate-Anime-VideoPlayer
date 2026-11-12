#include "app/play/backend_library.h"

#include <cstdint>
#include <format>
#include <mutex>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/utf8_path.h"

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
  // Un module sans point d'entrée ou d'une autre ABI n'arrête pas la recherche : le candidat suivant
  // s'essaie (A21-05). Sans aucun module valable, AbiMismatch l'emporte, pour que la réinstallation d'un
  // module périmé reste proposée.
  RefusalCode code = RefusalCode::LibraryUnavailable;
  for (const auto& path : candidates) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error)) {
      tried += std::format("\n  {} : absent", core::to_utf8(path));
      continue;
    }
    {
      Kept& k = kept();
      const std::scoped_lock lock(k.mutex);
      for (const auto& held : k.libraries)
        if (held->path_ == path && !held->replaced_on_disk()) return held;  // même chemin que open l'a reçu
    }
    std::shared_ptr<BackendLibrary> library(new BackendLibrary());
    // La date et la taille du fichier avant le chargement : un module remplacé pendant open est vu comme
    // remplacé, jamais comme celui qui est en mémoire (B21-04).
    const std::filesystem::file_time_type stamp = std::filesystem::last_write_time(path, error);
    const std::uintmax_t size = std::filesystem::file_size(path, error);
    // dlopen sous Linux et macOS, LoadLibraryExW sous Windows (dépendances cherchées à côté).
    auto opened = DynamicLibrary::open(std::filesystem::absolute(path, error));
    if (!opened) {
      tried += std::format("\n  {}", opened.error().detail);
      continue;
    }
    library->handle_.emplace(std::move(*opened));
    library->path_ = path;
    library->stamp_ = stamp;
    library->size_ = size;
    const DynamicLibrary& handle = *library->handle_;
    const auto abi = symbol<gpu::BackendAbiFn>(handle, "ajp_backend_abi");
    const auto layout = symbol<gpu::BackendLayoutFn>(handle, "ajp_backend_layout");
    library->create_ = symbol<gpu::BackendCreateFn>(handle, "ajp_backend_create");
    library->destroy_ = symbol<gpu::BackendDestroyFn>(handle, "ajp_backend_destroy");
    if (abi == nullptr || layout == nullptr || library->create_ == nullptr || library->destroy_ == nullptr) {
      tried += std::format("\n  {} : point d'entrée absent", core::to_utf8(path));
      continue;
    }
    if (auto ok = gpu::check_backend_abi(abi(), layout()); !ok) {
      tried += std::format("\n  {} : {}", core::to_utf8(path), ok.error().detail);
      code = RefusalCode::AbiMismatch;
      continue;
    }
    return library;
  }
  if (candidates.size() == 1 && tried.size() > 3) {
    // Un seul candidat : son refus tel quel, sans l'en-tête de la liste.
    return refuse(code, tried.substr(3));
  }
  return refuse(code, std::format("aucune bibliothèque de backend :{}", tried));
}

BackendLibrary::~BackendLibrary() = default;

bool BackendLibrary::replaced_on_disk() const {
  // La date seule ne suffit pas (C21-03) : tar remet celle de l'archive, et deux modules de même date
  // passaient pour le même. La taille s'y ajoute ; deux modules de même date et de même taille restent
  // possibles, mais il y faut deux constructions datées pareil et de la même longueur à l'octet.
  std::error_code error;
  const auto now = std::filesystem::last_write_time(path_, error);
  if (error) return false;
  const std::uintmax_t size = std::filesystem::file_size(path_, error);
  return now != stamp_ || (!error && size != size_);
}

void BackendLibrary::Destroy::operator()(gpu::Backend* backend) const {
  if (backend != nullptr) destroy(backend);
}

std::expected<std::shared_ptr<BackendLibrary>, Refusal> BackendLibrary::choose(
    std::span<const std::filesystem::path> candidates, const gpu::BackendInit& init) {
  std::string tried;
  // C21-02 : comme open, un module d'une autre ABI l'emporte sur l'absence quand rien n'a abouti, pour que
  // l'écran propose de réinstaller le module périmé au lieu de dire qu'il manque.
  RefusalCode code = RefusalCode::LibraryUnavailable;
  for (const auto& path : candidates) {
    auto library = open(std::span(&path, 1));
    if (!library) {
      tried += std::format("\n  {}", library.error().detail);
      if (library.error().code == RefusalCode::AbiMismatch) code = RefusalCode::AbiMismatch;
      continue;
    }
    if (auto probe = create(*library, init); !probe) {
      tried += std::format("\n  {} : {} ({})", core::to_utf8(path), probe.error().detail,
                           name(probe.error().code));
      continue;
    }
    (*library)->passed_over_ = std::move(tried);
    release_kept();
    return library;
  }
  release_kept();
  return refuse(code, std::format("aucun backend pour cet appareil :{}", tried));
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
