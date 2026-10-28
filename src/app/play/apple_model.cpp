#include "app/play/apple_model.h"

#include <algorithm>
#include <cstddef>
#include <format>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <vector>

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
  std::error_code error;
  std::filesystem::create_directories(dir, error);
  std::filesystem::path temp = target;
  temp += ".tmp";
  {
    std::ofstream out(temp, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(wanted.data()), static_cast<std::streamsize>(wanted.size()));
    if (!out.flush()) return std::unexpected(Refusal{RefusalCode::FileUnwritable, temp.string()});
  }
  std::filesystem::rename(temp, target, error);
  if (error) {
    return std::unexpected(
        Refusal{RefusalCode::FileUnwritable, std::format("{} : {}", target.string(), error.message())});
  }
  return target;
}

}  // namespace ajp::app::play
