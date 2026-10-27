#include "app/play/drba_model.h"

#include <format>
#include <string_view>
#include <system_error>
#include <vector>

#include "app/play/derived_model.h"
#include "prepare/onnx_dims.h"

namespace ajp::app::play {

std::optional<DrbaModel> find_distil_drba(std::span<const std::filesystem::path> dirs, bool prefer_half) {
  std::error_code error;
  for (const bool half : {prefer_half, !prefer_half}) {
    for (const std::filesystem::path& dir : dirs) {
      std::filesystem::path found = dir / (half ? kDistilDrbaHalfFile : kDistilDrbaFile);
      if (std::filesystem::is_regular_file(found, error))
        return DrbaModel{.path = std::move(found), .align = half ? kDistilDrbaHalfAlign : 64};
    }
  }
  return std::nullopt;
}

bool drba_half_flow(int work_height) {
  return work_height >= 1440;
}

std::expected<std::filesystem::path, Refusal> static_distil_drba(const std::filesystem::path& model,
                                                                 int padded_width, int padded_height,
                                                                 const std::filesystem::path& dir) {
  const std::vector<prepare::NamedDim> dims{{"H", padded_height}, {"W", padded_width}};
  const auto fix = [&](std::span<const std::byte> bytes) { return prepare::with_fixed_dims(bytes, dims); };
  return derived_model(model, std::format("{}x{}", padded_width, padded_height), dir, fix);
}

}  // namespace ajp::app::play
