// Sondage des bibliothèques du système : le chargeur Vulkan, le pilote CUDA,
// ROCm, OpenVINO. À la différence de vendor_hdr::DynamicLibrary, qui n'ouvre
// qu'un chemin absolu choisi par l'utilisateur, on cherche ici par nom, comme
// le système le ferait, pour savoir ce qui est installé.
//
// Une interface, pour que les tests remplacent le système par des sondes factices.
#pragma once

#include <expected>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>

#include "core/types.h"

namespace ajp::app {

class LoadedLibrary {
 public:
  virtual ~LoadedLibrary() = default;
  // Adresse du symbole, nullptr s'il manque.
  virtual void* raw_symbol(const char* name) const = 0;

  // Fn est un type de fonction : symbol<PFN>("cuDriverGetVersion") rend un Fn*, nullptr s'il manque.
  template <class Fn>
  Fn* symbol(const char* name) const {
    static_assert(std::is_function_v<Fn> && sizeof(Fn*) == sizeof(void*));
    // void* vers pointeur de fonction : admis sous condition par C++, garanti par POSIX
    // (dlsym) et par Windows (GetProcAddress).
    return reinterpret_cast<Fn*>(raw_symbol(name));
  }
};

class LibraryLoader {
 public:
  virtual ~LibraryLoader() = default;
  // Cherche name dans les dossiers du système. Refus LibraryUnavailable, avec le
  // message du chargeur, si elle n'y est pas ou ne se charge pas.
  virtual std::expected<std::unique_ptr<LoadedLibrary>, Refusal> open(std::string_view name) const = 0;
};

// dlopen(RTLD_LAZY | RTLD_LOCAL) sous Linux et macOS ; LoadLibraryExW avec les seuls
// dossiers du système sous Windows (jamais le dossier courant).
std::unique_ptr<LibraryLoader> system_library_loader();

}  // namespace ajp::app
