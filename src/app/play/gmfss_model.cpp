#include "app/play/gmfss_model.h"

#include <format>
#include <string>
#include <system_error>
#include <utility>

#include "app/play/derived_model.h"
#include "core/utf8_path.h"

namespace kirari::app::play {

std::optional<GmfssModels> find_gmfss(std::span<const std::filesystem::path> dirs) {
  std::error_code error;
  for (const std::filesystem::path& dir : dirs) {
    GmfssModels found{.pair = dir / kGmfssPairFile, .fusion = dir / kGmfssFusionFile};
    if (std::filesystem::is_regular_file(found.pair, error) &&
        std::filesystem::is_regular_file(found.fusion, error)) {
      if (std::filesystem::is_regular_file(dir / kGmfssHalfPairFile, error))
        found.half_pair = dir / kGmfssHalfPairFile;
      return found;
    }
  }
  return std::nullopt;
}

GmfssModels gmfss_chosen(const GmfssModels& found, bool half) {
  return {.pair = half && !found.half_pair.empty() ? found.half_pair : found.pair, .fusion = found.fusion};
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
  GmfssModels fixed;
  for (const auto& [from, into] :
       {std::pair{&models.pair, &fixed.pair}, std::pair{&models.fusion, &fixed.fusion}}) {
    auto derived = derived_model(*from, size, dir, fix);
    if (!derived) return std::unexpected(std::move(derived.error()));
    // Une taille par définition de vidéo lue : les anciennes s'en vont (A09-11).
    purge_sized_derivatives(dir, core::to_utf8(from->stem()) + "-", *derived);
    *into = std::move(*derived);
  }
  return fixed;
}

}  // namespace kirari::app::play
