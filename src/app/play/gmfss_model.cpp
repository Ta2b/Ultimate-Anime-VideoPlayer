#include "app/play/gmfss_model.h"

#include <format>
#include <system_error>

#include "app/play/derived_model.h"
#include "kernels/spec/rife.h"

namespace ajp::app::play {

std::optional<GmfssModels> find_gmfss(std::span<const std::filesystem::path> dirs, GmfssVariant variant) {
  std::error_code error;
  for (const bool half_flow : {true, false}) {
    const GmfssFiles files = gmfss_files(variant, half_flow);
    for (const std::filesystem::path& dir : dirs) {
      GmfssModels found{.pair = dir / files.pair,
                        .fusion = dir / files.fusion,
                        .align = half_flow ? kGmfssHalfAlign : kernels::kRifeAlign};
      if (std::filesystem::is_regular_file(found.pair, error) &&
          std::filesystem::is_regular_file(found.fusion, error))
        return found;
    }
  }
  return std::nullopt;
}

std::vector<prepare::NamedDim> gmfss_dims(int padded_width, int padded_height) {
  std::vector<prepare::NamedDim> dims{{"H", padded_height}, {"W", padded_width}};
  for (const int scale : {2, 4, 8}) {
    dims.push_back({std::format("H/{}", scale), padded_height / scale});
    dims.push_back({std::format("W/{}", scale), padded_width / scale});
  }
  return dims;
}

std::expected<GmfssModels, Refusal> static_gmfss(const GmfssModels& models, int padded_width,
                                                 int padded_height, const std::filesystem::path& dir) {
  const std::vector<prepare::NamedDim> dims = gmfss_dims(padded_width, padded_height);
  const auto fix = [&](std::span<const std::byte> model) { return prepare::with_fixed_dims(model, dims); };
  const std::string size = std::format("{}x{}", padded_width, padded_height);
  auto pair = derived_model(models.pair, size, dir, fix);
  if (!pair) return std::unexpected(std::move(pair.error()));
  auto fusion = derived_model(models.fusion, size, dir, fix);
  if (!fusion) return std::unexpected(std::move(fusion.error()));
  return GmfssModels{.pair = std::move(*pair), .fusion = std::move(*fusion), .align = models.align};
}

}  // namespace ajp::app::play
