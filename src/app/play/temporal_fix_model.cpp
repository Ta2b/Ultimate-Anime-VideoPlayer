#include "app/play/temporal_fix_model.h"

#include <format>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "app/play/derived_model.h"
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

std::expected<std::filesystem::path, Refusal> temporal_fix_graph(const std::filesystem::path& model,
                                                                 bool half,
                                                                 const std::filesystem::path& dir) {
  const auto derive = [half](std::span<const std::byte> bytes) {
    return prepare::with_frame_inputs(bytes, prepare::FrameSplit{.input = kTemporalFixInput,
                                                                 .frames = kTemporalFixFrames,
                                                                 .half = half,
                                                                 .center = kTemporalFixRadius});
  };
  return derived_model(model, half ? "frames-half" : "frames", dir, derive);
}

}  // namespace ajp::app::play
