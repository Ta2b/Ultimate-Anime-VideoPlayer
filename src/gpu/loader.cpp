#include "gpu/loader.h"

#include <algorithm>
#include <format>
#include <system_error>
#include <type_traits>

#include "vendor_hdr/path_text.h"

namespace ajp::gpu {
namespace {

constexpr std::string_view kPrefix = "ajp_backend_";
#ifdef _WIN32
constexpr std::string_view kSuffix = ".dll";
#else
constexpr std::string_view kSuffix = ".so";
#endif

// DynamicLibrary::symbol attend le type de la fonction, pas celui de son adresse.
template <class Pointer>
using Function = std::remove_pointer_t<Pointer>;

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

// Un nom de backend devient un nom de fichier : ni séparateur, ni point, ni majuscule.
bool valid_name(std::string_view name) {
  const auto allowed = [](char c) { return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_'; };
  return !name.empty() && name.size() <= 64 && std::ranges::all_of(name, allowed);
}

struct AbiStamp {
  int abi = 0;
  std::uint64_t layout = 0;
};

// Version et empreinte binaire de la bibliothèque, lues avant tout autre appel.
std::expected<AbiStamp, Refusal> read_abi(const vendor_hdr::DynamicLibrary& library) {
  auto abi = library.symbol<Function<BackendAbiFn>>("ajp_backend_abi");
  if (!abi) return std::unexpected(std::move(abi.error()));
  auto layout = library.symbol<Function<BackendLayoutFn>>("ajp_backend_layout");
  if (!layout) return std::unexpected(std::move(layout.error()));
  return AbiStamp{.abi = (*abi)(), .layout = (*layout)()};
}

std::expected<vendor_hdr::DynamicLibrary, Refusal> open_file(const std::filesystem::path& path) {
  std::error_code error;
  const std::filesystem::path absolute = std::filesystem::absolute(path, error);
  if (error || !std::filesystem::is_regular_file(absolute, error)) {
    return refuse(RefusalCode::LibraryUnavailable,
                  std::format("{} : fichier absent", vendor_hdr::utf8(path)));
  }
  return vendor_hdr::DynamicLibrary::open(absolute);
}

// Ouvre la bibliothèque le temps de lire son ABI, sans créer de backend.
BackendListing probe(std::string name, const std::filesystem::path& path) {
  BackendListing listing{.name = std::move(name), .path = path, .abi = 0, .usable = {}};
  auto library = open_file(path);
  auto stamp = library ? read_abi(*library) : std::unexpected(std::move(library.error()));
  if (!stamp) {
    listing.abi = std::unexpected(stamp.error());
    listing.usable = std::unexpected(std::move(stamp.error()));
    return listing;
  }
  listing.abi = stamp->abi;
  listing.usable = check_backend_abi(stamp->abi, stamp->layout);
  return listing;
}

}  // namespace

std::string backend_file_name(std::string_view name) {
  return std::format("{}{}{}", kPrefix, name, kSuffix);
}

std::expected<LoadedBackend, Refusal> LoadedBackend::load(const std::filesystem::path& dir,
                                                          std::string_view name, const BackendInit& init) {
  if (!valid_name(name)) {
    return refuse(RefusalCode::FieldOutOfRange, std::format("backend « {} » : nom hors de [a-z0-9_]", name));
  }
  return load_file(dir / backend_file_name(name), init);
}

std::expected<LoadedBackend, Refusal> LoadedBackend::load_file(const std::filesystem::path& path,
                                                               const BackendInit& init) {
  auto library = open_file(path);
  if (!library) return std::unexpected(std::move(library.error()));
  const std::string shown = vendor_hdr::utf8(path.filename());

  // L'ABI se lit avant tout appel qui passerait un BackendInit d'une autre forme.
  auto stamp = read_abi(*library);
  if (!stamp) return std::unexpected(std::move(stamp.error()));
  if (auto ok = check_backend_abi(stamp->abi, stamp->layout); !ok) {
    return refuse(RefusalCode::AbiMismatch, std::format("{} : {}", shown, ok.error().detail));
  }
  auto create = library->symbol<Function<BackendCreateFn>>("ajp_backend_create");
  if (!create) return std::unexpected(std::move(create.error()));
  auto destroy = library->symbol<Function<BackendDestroyFn>>("ajp_backend_destroy");
  if (!destroy) return std::unexpected(std::move(destroy.error()));

  Refusal refusal{.code = RefusalCode::LibraryUnavailable, .detail = {}};
  BackendInit with_refusal = init;
  with_refusal.refusal = &refusal;
  Backend* backend = (*create)(kBackendAbi, &with_refusal);
  if (backend == nullptr) {
    if (refusal.detail.empty()) {
      refusal.detail = std::format("{} : ajp_backend_create a rendu nullptr sans dire pourquoi", shown);
    }
    return std::unexpected(std::move(refusal));
  }
  // Facultatif : un backend sans noyaux d'entrée de RIFE ne sert qu'aux plans sans interpolation.
  RifeInputOps* rife = nullptr;
  if (auto rife_input = library->symbol<Function<BackendRifeInputFn>>("ajp_backend_rife_input")) {
    rife = (*rife_input)(backend);
  }
  return LoadedBackend(std::move(*library), backend, *destroy, rife);
}

std::expected<std::vector<BackendListing>, Refusal> list_backends(const std::filesystem::path& dir) {
  std::error_code error;
  std::filesystem::directory_iterator it(dir, error);
  if (error) {
    return refuse(RefusalCode::FileUnreadable,
                  std::format("{} : {}", vendor_hdr::utf8(dir), error.message()));
  }
  std::vector<BackendListing> found;
  for (; it != std::filesystem::directory_iterator{}; it.increment(error)) {
    if (error) {
      return refuse(RefusalCode::FileUnreadable,
                    std::format("{} : {}", vendor_hdr::utf8(dir), error.message()));
    }
    const std::string file = vendor_hdr::utf8(it->path().filename());
    if (file.size() <= kPrefix.size() + kSuffix.size() || !file.starts_with(kPrefix) ||
        !file.ends_with(kSuffix)) {
      continue;
    }
    std::string name = file.substr(kPrefix.size(), file.size() - kPrefix.size() - kSuffix.size());
    if (!valid_name(name)) continue;
    found.push_back(probe(std::move(name), it->path()));
  }
  std::ranges::sort(found, {}, &BackendListing::name);
  return found;
}

}  // namespace ajp::gpu
