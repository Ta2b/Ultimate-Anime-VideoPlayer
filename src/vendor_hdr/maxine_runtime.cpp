#include "vendor_hdr/maxine_runtime.h"

#include <algorithm>
#include <format>
#include <optional>
#include <system_error>
#include <utility>

#include "vendor_hdr/log.h"
#include "vendor_hdr/path_text.h"
#include "vendor_hdr/sdk_layout.h"

namespace ajp::vendor_hdr {
namespace {

namespace fs = std::filesystem;

struct Candidate {
  fs::path relative;
  bool required = false;
  Refusal failure{RefusalCode::LibraryUnavailable, {}};  // dernier échec d'ouverture
};

// Dépendances d'abord (external/), puis le cœur (lib/), puis la fonction TrueHDR.
int load_rank(const fs::path& relative) {
  const std::string first = utf8(*relative.begin());
  return first == "external" ? 0 : first == "lib" ? 1 : 2;
}

// Les bibliothèques à ouvrir, chacune une fois : un alias (libX.so -> libX.so.12) vise le même
// fichier réel que son nom versionné.
class CandidateList {
 public:
  explicit CandidateList(fs::path root) : root_(std::move(root)) {}

  // false si relative ne désigne pas un fichier (absent, lien pendant, dossier).
  bool add(const fs::path& relative, bool required) {
    std::error_code error;
    const fs::path real = fs::canonical(root_ / relative, error);
    if (error || !fs::is_regular_file(real, error)) return false;
    if (std::ranges::find(real_files_, real) == real_files_.end()) {
      real_files_.push_back(real);
      list_.push_back(Candidate{.relative = relative, .required = required});
    }
    return true;
  }

  std::vector<Candidate> take() && {
    std::ranges::stable_sort(list_, {}, [](const Candidate& c) { return load_rank(c.relative); });
    return std::move(list_);
  }

 private:
  fs::path root_;
  std::vector<fs::path> real_files_;
  std::vector<Candidate> list_;
};

std::expected<std::vector<Candidate>, Refusal> candidates(const fs::path& root) {
  CandidateList list(root);
  for (const RequiredFile& required : layout(SdkKind::MaxineVfx).required) {
    if (required.format != BinaryFormat::ElfX86_64) continue;
    const std::optional<fs::path> relative = locate(root, required);
    if (!relative || !list.add(*relative, true)) {
      return refuse(RefusalCode::LibraryUnavailable, std::format("{} absent du SDK", required.path));
    }
  }
  const fs::path feature(kMaxineFeatureLibraries);
  for (const fs::path& relative : loadable_libraries(SdkKind::MaxineVfx, root)) {
    list.add(relative, relative.parent_path() == feature);
  }
  return std::move(list).take();
}

}  // namespace

const DynamicLibrary* find_library(const std::vector<LoadedLibrary>& loaded, std::string_view relative) {
  const auto found = std::ranges::find(loaded, fs::path(relative), &LoadedLibrary::relative);
  return found == loaded.end() ? nullptr : &found->library;
}

std::expected<std::vector<LoadedLibrary>, Refusal> load_maxine_runtime(const fs::path& root) {
  auto pending = candidates(root);
  if (!pending) return std::unexpected(pending.error());

  std::vector<LoadedLibrary> loaded;
  for (bool progress = true; progress && !pending->empty();) {
    progress = false;
    std::vector<Candidate> retry;
    for (Candidate& candidate : *pending) {
      const SymbolScope scope = load_rank(candidate.relative) == 2 ? SymbolScope::Global : SymbolScope::Local;
      auto library = DynamicLibrary::open(root / candidate.relative, scope);
      if (!library) {
        candidate.failure = library.error();
        retry.push_back(std::move(candidate));
        continue;
      }
      loaded.push_back(
          LoadedLibrary{.relative = std::move(candidate.relative), .library = std::move(*library)});
      progress = true;
    }
    *pending = std::move(retry);
  }
  for (const Candidate& candidate : *pending) {
    if (candidate.required) return std::unexpected(candidate.failure);
    make_logger()->info("Maxine : {} non chargée, ignorée ({})", utf8(candidate.relative),
                        candidate.failure.detail);
  }
  return loaded;
}

}  // namespace ajp::vendor_hdr
