#include "vendor_hdr/maxine_runtime.h"

#include <algorithm>
#include <format>
#include <system_error>
#include <utility>

#include "vendor_hdr/sdk_layout.h"

namespace ajp::vendor_hdr {

namespace fs = std::filesystem;

const DynamicLibrary* find_library(const std::vector<LoadedLibrary>& loaded, std::string_view relative) {
  const auto found = std::ranges::find(loaded, fs::path(relative), &LoadedLibrary::relative);
  return found == loaded.end() ? nullptr : &found->library;
}

std::expected<std::vector<LoadedLibrary>, Refusal> load_maxine_runtime(const fs::path& root) {
  std::vector<LoadedLibrary> loaded;
  for (const std::string_view relative : maxine_runtime()) {
    // Un lien pendant ou un dossier n'est pas une bibliothèque : refus nommé avant dlopen.
    std::error_code error;
    if (!fs::is_regular_file(root / relative, error)) {
      return refuse(RefusalCode::LibraryUnavailable, std::format("{} absent du SDK", relative));
    }
    const SymbolScope scope =
        relative.starts_with(kMaxineFeatureLibraries) ? SymbolScope::Global : SymbolScope::Local;
    auto library = DynamicLibrary::open(root / relative, scope);
    if (!library) return std::unexpected(library.error());  // loaded se ferme en sortant
    loaded.push_back(LoadedLibrary{.relative = fs::path(relative), .library = std::move(*library)});
  }
  return loaded;
}

}  // namespace ajp::vendor_hdr
