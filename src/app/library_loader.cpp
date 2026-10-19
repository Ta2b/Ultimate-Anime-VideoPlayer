#include "app/library_loader.h"

#include <format>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ajp::app {
namespace {

#if defined(_WIN32)

class WindowsLibrary final : public LoadedLibrary {
 public:
  explicit WindowsLibrary(HMODULE module) : module_(module) {}
  WindowsLibrary(const WindowsLibrary&) = delete;
  WindowsLibrary& operator=(const WindowsLibrary&) = delete;
  ~WindowsLibrary() override { FreeLibrary(module_); }
  void* raw_symbol(const char* name) const override {
    return reinterpret_cast<void*>(GetProcAddress(module_, name));
  }

 private:
  HMODULE module_;
};

class SystemLoader final : public LibraryLoader {
 public:
  std::expected<std::unique_ptr<LoadedLibrary>, Refusal> open(std::string_view name) const override {
    const std::wstring wide(name.begin(), name.end());  // noms ASCII de la liste des sondes
    HMODULE module = LoadLibraryExW(wide.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (module == nullptr) {
      return std::unexpected(Refusal{RefusalCode::LibraryUnavailable,
                                     std::format("{} : erreur Windows {}", name, GetLastError())});
    }
    return std::make_unique<WindowsLibrary>(module);
  }
};

#else

class PosixLibrary final : public LoadedLibrary {
 public:
  explicit PosixLibrary(void* handle) : handle_(handle) {}
  PosixLibrary(const PosixLibrary&) = delete;
  PosixLibrary& operator=(const PosixLibrary&) = delete;
  ~PosixLibrary() override { dlclose(handle_); }
  void* raw_symbol(const char* name) const override { return dlsym(handle_, name); }

 private:
  void* handle_;
};

class SystemLoader final : public LibraryLoader {
 public:
  std::expected<std::unique_ptr<LoadedLibrary>, Refusal> open(std::string_view name) const override {
    const std::string file(name);
    void* handle = dlopen(file.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (handle == nullptr) {
      const char* error = dlerror();
      return std::unexpected(Refusal{RefusalCode::LibraryUnavailable,
                                     error != nullptr ? std::string(error) : file + " introuvable"});
    }
    return std::make_unique<PosixLibrary>(handle);
  }
};

#endif

}  // namespace

std::unique_ptr<LibraryLoader> system_library_loader() {
  return std::make_unique<SystemLoader>();
}

}  // namespace ajp::app
