// Le modèle de l'interpolation d'Apple (apple_model.h) : écrit une fois à l'interface de RIFE v2
// avec la marque de VTFrameRateConversion, réécrit s'il a été abîmé, laissé tel quel sinon.
#include "app/play/apple_model.h"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/system_interpolator.h"
#include "prepare/onnx_scan.h"

namespace ajp::app::play {
namespace {

std::vector<std::byte> read_all(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  const std::vector<char> chars{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  std::vector<std::byte> bytes(chars.size());
  for (std::size_t i = 0; i < chars.size(); ++i) bytes[i] = static_cast<std::byte>(chars[i]);
  return bytes;
}

}  // namespace

TEST_CASE("apple_model : écrit une fois, réécrit s'il est abîmé", "[app][apple]") {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "ajp_apple_model_test";
  std::filesystem::remove_all(dir);
  auto written = apple_model(dir);
  REQUIRE(written);
  CHECK(written->filename() == std::string(gpu::kVideoToolboxFrameRate) + ".onnx");
  const std::vector<std::byte> first = read_all(*written);
  auto summary = prepare::scan_onnx(first);
  REQUIRE(summary);
  REQUIRE(summary->inputs.size() == 1);
  CHECK(summary->inputs[0].name == "input");
  CHECK(summary->inputs[0].elem_type == static_cast<std::int32_t>(prepare::OnnxType::Float16));
  CHECK(summary->inputs[0].dims == std::vector<std::int64_t>{1, 7, -1, -1});

  const auto stamp = std::filesystem::last_write_time(*written);
  REQUIRE(apple_model(dir));
  CHECK(std::filesystem::last_write_time(*written) == stamp);  // déjà là : rien d'écrit

  std::ofstream(*written, std::ios::binary | std::ios::trunc) << "abîmé";
  REQUIRE(apple_model(dir));
  CHECK(read_all(*written) == first);
  std::filesystem::remove_all(dir);
}

}  // namespace ajp::app::play
