#include "core/dynamic_library.h"

#ifdef _WIN32
#include <mutex>
#include <set>
#include <string>
#include <system_error>
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ajp {
namespace {

std::string utf8(const std::filesystem::path& path) {
  return library_path_text(path);
}

#ifdef _WIN32
// Chaque dossier une fois dans la recherche du processus (B20-03) : AddDllDirectory ajoute une entrée à
// chaque appel, la liste grossissait d'un doublon par bibliothèque ouverte.
void add_dll_directory_once(const std::filesystem::path& dir) {
  static std::mutex mutex;
  static std::set<std::filesystem::path> added;
  const std::scoped_lock lock(mutex);
  if (!added.insert(dir).second) return;
  if (AddDllDirectory(dir.c_str()) == nullptr) added.erase(dir);
}

// Le code seul ne dit rien à distance : 126 cache quelle dépendance manque, 193 une DLL x86 ou abîmée
// (C20-04).
std::string windows_error(DWORD code) {
  std::string text = std::system_category().message(static_cast<int>(code));
  while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ')) text.pop_back();
  std::string hint;
  if (code == ERROR_BAD_EXE_FORMAT) hint = ", DLL 32 bits ou abîmée";
  return std::format("erreur Windows {} ({}{})", code, text, hint);
}

// 126 : une dépendance manque ; elle se cherche dans ces dossiers.
std::string not_found_note(DWORD code, const std::filesystem::path& path) {
  if (code != ERROR_MOD_NOT_FOUND) return {};
  return std::format(", une dépendance manque, cherchée dans {} puis les dossiers du système",
                     library_path_text(path.parent_path()));
}
#endif

}  // namespace

std::expected<DynamicLibrary, Refusal> DynamicLibrary::open(const std::filesystem::path& given,
                                                            SymbolScope scope) {
  // « C:\x\engines/trt/nvinfer_10.dll » : des séparateurs mêlés trompent le chargeur sur le dossier de la
  // DLL, ses voisines ne se trouvent plus (C20-05).
  const std::filesystem::path path = std::filesystem::path(given).lexically_normal().make_preferred();
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
  // Les dépendances se cherchent à côté de la DLL, puis dans les dossiers du système. Son dossier s'ajoute
  // aussi à la recherche du processus : une DLL qu'elle charge plus tard par son nom s'y trouve, alors que le
  // PATH n'est plus lu (SetDefaultDllDirectories dans main.cpp, A20-07). Jamais retiré : la bibliothèque peut
  // charger jusqu'à sa fermeture, et un dossier du lecteur n'apporte rien d'étranger.
  add_dll_directory_once(path.parent_path());
  HMODULE module = LoadLibraryExW(path.c_str(), nullptr,
                                  LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
  if (module == nullptr) {
    const DWORD code = GetLastError();
    return refuse(
        RefusalCode::LibraryUnavailable,
        std::format("{} ne se charge pas : {}{}", shown, windows_error(code), not_found_note(code, path)));
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

std::expected<DynamicLibrary, Refusal> DynamicLibrary::open_system(const char* name) {
  const std::filesystem::path path(name);
#ifdef _WIN32
  // System32 seul (B20-03) : LOAD_LIBRARY_SEARCH_DEFAULT_DIRS chercherait aussi le dossier du lecteur et ceux
  // des moteurs déjà ouverts, où un vulkan-1.dll périmé copié à la main masquerait le chargeur du pilote.
  HMODULE module = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (module == nullptr) {
    return refuse(RefusalCode::LibraryUnavailable,
                  std::format("{} ne se charge pas : {}", name, windows_error(GetLastError())));
  }
  return DynamicLibrary(static_cast<void*>(module), path);
#else
  void* handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    const char* error = dlerror();
    return refuse(RefusalCode::LibraryUnavailable, std::format("{} ne se charge pas : {}", name,
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

}  // namespace ajp
