#include "app/play/gmfss_model.h"

#include <format>
#include <system_error>

#include "app/play/derived_model.h"
#include "kernels/spec/rife.h"

namespace ajp::app::play {

std::optional<GmfssModels> find_gmfss(std::span<const std::filesystem::path> dirs, GmfssVariant variant,
                                      bool prefer_half, PrecisionTier tier) {
  std::error_code error;
  // Les essais dans l'ordre : la paire du cran demandé (demi-échelle), puis les graphes du flot à
  // demi-échelle et du flot entier, dans l'ordre que prefer_half donne.
  struct Attempt {
    bool half_flow;
    PrecisionTier tier;
  };
  std::vector<Attempt> attempts;
  if (tier != PrecisionTier::Balanced) attempts.push_back({true, tier});
  for (const bool half_flow : {prefer_half, !prefer_half})
    attempts.push_back({half_flow, PrecisionTier::Balanced});
  for (const auto& [half_flow, attempt_tier] : attempts) {
    const GmfssFiles files = gmfss_files(variant, half_flow, attempt_tier);
    for (const std::filesystem::path& dir : dirs) {
      GmfssModels found{.pair = dir / files.pair,
                        .fusion = dir / files.fusion,
                        .align = half_flow ? kGmfssHalfAlign : kernels::kRifeAlign,
                        .tier = attempt_tier};
      if (std::filesystem::is_regular_file(found.pair, error) &&
          std::filesystem::is_regular_file(found.fusion, error))
        return found;
    }
  }
  return std::nullopt;
}

bool gmfss_tier_available(std::span<const std::filesystem::path> dirs, GmfssVariant variant,
                          PrecisionTier tier) {
  const std::optional<GmfssModels> found = find_gmfss(dirs, variant, true, tier);
  return found && found->tier == tier;
}

TierSupport gmfss_tiers(std::span<const std::filesystem::path> dirs, GmfssVariant variant, bool fp8_engine) {
  return TierSupport{
      .performance = fp8_engine && gmfss_tier_available(dirs, variant, PrecisionTier::Performance),
      .quality = gmfss_tier_available(dirs, variant, PrecisionTier::Quality)};
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
