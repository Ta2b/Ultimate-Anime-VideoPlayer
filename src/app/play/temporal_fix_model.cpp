#include "app/play/temporal_fix_model.h"

#include <format>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "app/play/derived_model.h"
#include "core/utf8_path.h"
#include "prepare/onnx_dims.h"
#include "prepare/onnx_frames.h"

namespace ajp::app::play {

std::optional<std::filesystem::path> find_temporal_fix(TemporalFix fix,
                                                       std::span<const std::filesystem::path> dirs) {
  const std::string_view file = level_of(fix).model;
  if (file.empty()) return std::nullopt;
  std::error_code error;
  for (const std::filesystem::path& dir : dirs) {
    std::filesystem::path candidate = dir / file;
    if (std::filesystem::is_regular_file(candidate, error)) return candidate;
  }
  return std::nullopt;
}

std::expected<std::filesystem::path, Refusal> static_temporal_fix(const std::filesystem::path& model,
                                                                  int width, int height,
                                                                  const std::filesystem::path& dir) {
  if (width <= 0 || height <= 0 || width % 2 != 0 || height % 2 != 0) {
    return refuse(
        RefusalCode::FieldOutOfRange,
        std::format("stabilisation temporelle : images de {}x{}, une taille paire attendue", width, height));
  }
  // Les graphes de pifroggi nomment leurs dimensions « height » et « width », à l'entrée comme à la sortie.
  const std::vector<prepare::NamedDim> dims{{"height", height}, {"width", width}};
  const auto derive =
      [&](std::span<const std::byte> bytes) -> std::expected<std::vector<std::byte>, Refusal> {
    auto fixed = prepare::with_fixed_dims(bytes, dims);
    if (!fixed) return fixed;
    return prepare::with_frame_inputs(*fixed, kTemporalFixInput, kTemporalFixFrames);
  };
  auto derived = derived_model(model, std::format("{}x{}-frames", width, height), dir, derive);
  // Une taille par définition de vidéo lue : les anciennes s'en vont (A09-11).
  if (derived) purge_sized_derivatives(dir, core::to_utf8(model.stem()) + "-", *derived);
  return derived;
}

}  // namespace ajp::app::play
