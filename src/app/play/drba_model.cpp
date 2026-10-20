#include "app/play/drba_model.h"

#include <format>
#include <system_error>
#include <vector>

#include "app/play/derived_model.h"
#include "prepare/onnx_dims.h"

namespace ajp::app::play {

std::optional<std::filesystem::path> find_distil_drba(std::span<const std::filesystem::path> dirs) {
  std::error_code error;
  for (const std::filesystem::path& dir : dirs) {
    std::filesystem::path found = dir / kDistilDrbaFile;
    if (std::filesystem::is_regular_file(found, error)) return found;
  }
  return std::nullopt;
}

std::expected<std::filesystem::path, Refusal> static_distil_drba(const std::filesystem::path& model,
                                                                 int padded_width, int padded_height,
                                                                 const std::filesystem::path& dir) {
  const std::vector<prepare::NamedDim> dims{{"H", padded_height}, {"W", padded_width}};
  const auto fix = [&](std::span<const std::byte> bytes) { return prepare::with_fixed_dims(bytes, dims); };
  return derived_model(model, std::format("{}x{}", padded_width, padded_height), dir, fix);
}

}  // namespace ajp::app::play
