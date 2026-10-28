#include "vendor_hdr/dynamic_library.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ajp::vendor_hdr {

std::expected<DynamicLibrary, Refusal> DynamicLibrary::open(const std::filesystem::path& path,
                                                            SymbolScope scope) {
  const std::string shown = utf8(path.filename());
  // Un nom seul ferait chercher le système (LD_LIBRARY_PATH, dossiers de Windows) : jamais.
  // LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR exige d'ailleurs un chemin absolu.
  if (!path.is_absolute()) {
    return refuse(RefusalCode::LibraryUnavailable,
                  std::format("{} : chemin relatif refusé, une bibliothèque s'ouvre par son chemin absolu",
                              utf8(path)));
  }
#ifdef _WIN32
  (void)scope;  // Windows n'a qu'un espace de symboles par module
  // Les dépendances se cherchent à côté de la DLL, puis dans les dossiers du système.
  HMODULE module = LoadLibraryExW(path.c_str(), nullptr,
                                  LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
  if (module == nullptr) {
    return refuse(RefusalCode::LibraryUnavailable,
                  std::format("{} ne se charge pas (erreur Windows {})", shown, GetLastError()));
  }
  return DynamicLibrary(static_cast<void*>(module), path);
#else
  const int flags = RTLD_NOW | (scope == SymbolScope::Global ? RTLD_GLOBAL : RTLD_LOCAL);
  void* handle = dlopen(path.c_str(), flags);
  if (handle == nullptr) {
    const char* error = dlerror();
    return refuse(RefusalCode::LibraryUnavailable, std::format("{} ne se charge pas : {}", shown,
                                                               error != nullptr ? error : "erreur inconnue"));
  }
  return DynamicLibrary(handle, path);
#endif
}

DynamicLibrary::DynamicLibrary(DynamicLibrary&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)), path_(std::move(other.path_)) {}

DynamicLibrary& DynamicLibrary::operator=(DynamicLibrary&& other) noexcept {
  if (this != &other) {
    close();
    handle_ = std::exchange(other.handle_, nullptr);
    path_ = std::move(other.path_);
  }
  return *this;
}

DynamicLibrary::~DynamicLibrary() {
  close();
}

void* DynamicLibrary::raw_symbol(const char* name) const {
  if (handle_ == nullptr) return nullptr;
#ifdef _WIN32
  return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
  return dlsym(handle_, name);
#endif
}

void DynamicLibrary::close() {
  if (handle_ == nullptr) return;
#ifdef _WIN32
  FreeLibrary(static_cast<HMODULE>(handle_));
#else
  dlclose(handle_);
#endif
  handle_ = nullptr;
}

}  // namespace ajp::vendor_hdr
