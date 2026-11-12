#include "app/play/invalid_copies.h"

#include <algorithm>
#include <string>
#include <system_error>
#include <vector>

#include "core/utf8_path.h"

namespace ajp::app::play {

void prune_invalid_copies(const std::filesystem::path& path, std::size_t kept) {
  const std::string prefix = core::to_utf8(path.filename()) + ".invalide-";
  const std::filesystem::path dir = path.has_parent_path() ? path.parent_path() : std::filesystem::path(".");
  std::vector<std::filesystem::path> copies;
  std::error_code error;
  for (std::filesystem::directory_iterator it(dir, error), end; !error && it != end; it.increment(error)) {
    if (core::to_utf8(it->path().filename()).starts_with(prefix)) copies.push_back(it->path());
  }
  if (copies.size() <= kept) return;
  std::ranges::sort(copies);  // AAAAMMJJ-HHMMSS : l'ordre des noms est celui des dates
  for (std::size_t i = 0; i + kept < copies.size(); ++i) {
    std::error_code gone;
    std::filesystem::remove(copies[i], gone);
  }
}

}  // namespace ajp::app::play
