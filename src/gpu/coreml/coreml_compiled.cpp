#include "gpu/coreml/coreml_compiled.h"

#include <cctype>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>

namespace ajp::gpu::coreml_rt {

std::string ml_program_name(std::string_view name) {
  std::string out;
  out.reserve(name.size() + 1);
  if (!name.empty() && std::isdigit(static_cast<unsigned char>(name.front())) != 0) out.push_back('_');
  for (const char c : name) {
    const bool kept = std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
    out.push_back(kept ? c : '_');
  }
  return out;
}

std::optional<std::filesystem::path> single_compiled(const std::filesystem::path& dir) {
  std::error_code error;
  std::optional<std::filesystem::path> found;
  auto it = std::filesystem::recursive_directory_iterator(dir, error);
  if (error) return std::nullopt;
  for (const auto end = std::filesystem::recursive_directory_iterator(); it != end; it.increment(error)) {
    if (error) return std::nullopt;
    if (!it->is_directory(error) || it->path().extension() != ".mlmodelc") continue;
    if (found) return std::nullopt;  // plusieurs morceaux
    found = it->path();
    it.disable_recursion_pending();  // le contenu d'un modèle compilé n'en contient pas d'autre
  }
  return found;
}

}  // namespace ajp::gpu::coreml_rt
