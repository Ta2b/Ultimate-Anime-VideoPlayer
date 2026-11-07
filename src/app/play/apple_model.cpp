#include "app/play/apple_model.h"

#include <algorithm>
#include <cstddef>
#include <format>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "core/file_sync.h"
#include "gpu/system_interpolator.h"
#include "prepare/onnx_blend.h"

namespace ajp::app::play {

std::expected<std::filesystem::path, Refusal> apple_model(const std::filesystem::path& dir) {
  const std::vector<std::byte> wanted =
      prepare::blend_model(gpu::kSystemInterpolatorProducer, gpu::kVideoToolboxFrameRate);
  std::filesystem::path target = dir / std::format("{}.onnx", gpu::kVideoToolboxFrameRate);
  {
    std::ifstream in(target, std::ios::binary);
    const std::vector<char> found{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    if (in.good() || in.eof()) {
      const auto same = [](char c, std::byte b) { return static_cast<std::byte>(c) == b; };
      if (std::ranges::equal(found, wanted, same)) return target;
    }
  }
  if (auto written = core::write_atomic(target, wanted); !written) return std::unexpected(written.error());
  return target;
}

}  // namespace ajp::app::play
