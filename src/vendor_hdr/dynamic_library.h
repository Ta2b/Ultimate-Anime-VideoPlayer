// Bibliothèque chargée à l'exécution : dlopen sous Linux, LoadLibraryExW sous Windows.
// Fermée à la destruction ; un symbole ne sert que tant qu'elle vit.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <string>
#include <type_traits>
#include <utility>

#include "vendor_hdr/refusal.h"

namespace ajp::vendor_hdr {

enum class SymbolScope : std::uint8_t {
  Local,   // RTLD_LOCAL : symboles réservés aux bibliothèques qui en dépendent
  Global,  // RTLD_GLOBAL : visibles des bibliothèques ouvertes ensuite (fonctions Maxine)
};

class DynamicLibrary {
 public:
  // Refuse (LibraryUnavailable) avec le message du chargeur du système.
  static std::expected<DynamicLibrary, Refusal> open(const std::filesystem::path& path,
                                                     SymbolScope scope = SymbolScope::Local);

  DynamicLibrary(DynamicLibrary&& other) noexcept;
  DynamicLibrary& operator=(DynamicLibrary&& other) noexcept;
  DynamicLibrary(const DynamicLibrary&) = delete;
  DynamicLibrary& operator=(const DynamicLibrary&) = delete;
  ~DynamicLibrary();

  const std::filesystem::path& path() const { return path_; }

  // Fn est un type de fonction : symbol<RunFn>("NvVFX_Run") rend un RunFn*.
  template <class Fn>
  std::expected<Fn*, Refusal> symbol(const char* name) const {
    void* address = raw_symbol(name);
    if (address == nullptr) {
      return refuse(RefusalCode::LibraryUnavailable,
                    std::format("symbole {} absent de {}", name, path_.filename().string()));
    }
    // void* vers pointeur de fonction : admis sous condition par C++, garanti par POSIX
    // (dlsym) et par Windows (GetProcAddress).
    static_assert(std::is_function_v<Fn> && sizeof(Fn*) == sizeof(void*));
    return reinterpret_cast<Fn*>(address);
  }

 private:
  DynamicLibrary(void* handle, std::filesystem::path path) : handle_(handle), path_(std::move(path)) {}
  void* raw_symbol(const char* name) const;
  void close();

  void* handle_ = nullptr;
  std::filesystem::path path_;
};

}  // namespace ajp::vendor_hdr
