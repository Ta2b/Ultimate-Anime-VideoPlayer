#include <array>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "gpu/winml/winml_detail.h"
#include "gpu/winml/winml_runtime.h"

namespace {

using ajp::RefusalCode;
using namespace ajp::gpu::winml_rt;

struct TempDir {
  explicit TempDir(const char* test)
      : path(std::filesystem::temp_directory_path() / std::format("ajp-winml-{}", test)) {
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  std::filesystem::path path;
};

void write_text(const std::filesystem::path& file, const std::string& text) {
  std::ofstream out(file, std::ios::binary | std::ios::trunc);
  out << text;
}

}  // namespace

TEST_CASE("winml : le manifeste relit ce qu'il écrit, chemin UTF-8 compris") {
  TempDir dir("manifest");
  const std::filesystem::path artifact = dir.path / "span-abc.winml";
  CHECK(cache_of(artifact).filename() == "span-abc.winml.mxcache");
  CHECK(onnx_of(artifact).filename() == "span-abc.winml.onnx");

  Manifest written;
  written.onnx = std::filesystem::path(u8"C:/Utilisateurs/Élodie/モデル/2x_AnimeJaNai.onnx");
  written.dims = {{"batch_size", 1}, {"height", 1080}, {"width with space", 1920}};
  written.io = true;
  written.fp16 = true;
  REQUIRE(write_manifest(artifact, written));

  auto read = read_manifest(artifact);
  REQUIRE(read);
  CHECK(read->onnx.u8string() == written.onnx.u8string());
  CHECK(read->dims == written.dims);
  CHECK(read->io);
  CHECK(read->fp16);

  Manifest plain;
  plain.onnx = dir.path / "plain.onnx";
  REQUIRE(write_manifest(artifact, plain));
  read = read_manifest(artifact);
  REQUIRE(read);
  CHECK(read->onnx == plain.onnx);
  CHECK(read->dims.empty());
  CHECK_FALSE(read->io);
  CHECK_FALSE(read->fp16);
}

TEST_CASE("winml : un manifeste absent ou abîmé se refuse en CacheIo") {
  TempDir dir("manifest-bad");
  auto missing = read_manifest(dir.path / "absent.winml");
  REQUIRE_FALSE(missing);
  CHECK(missing.error().code == RefusalCode::CacheIo);

  const std::filesystem::path no_onnx = dir.path / "sans-onnx.winml";
  write_text(no_onnx, "dim 1 batch\nfp16\n");
  auto refused = read_manifest(no_onnx);
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::CacheIo);

  const std::filesystem::path no_symbol = dir.path / "sans-symbole.winml";
  write_text(no_symbol, "onnx m.onnx\ndim 4\n");
  refused = read_manifest(no_symbol);
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::CacheIo);

  const std::filesystem::path bad_value = dir.path / "valeur.winml";
  write_text(bad_value, "onnx m.onnx\ndim x batch\n");
  refused = read_manifest(bad_value);
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::CacheIo);
}

TEST_CASE("winml : l'appareil de la chaîne d'abord, sinon le premier GPU AMD") {
  constexpr std::uint32_t amd = 0x1002;
  constexpr std::uint32_t nvidia = 0x10de;
  const std::array<DeviceId, 3> offered{{{nvidia, 0x2860}, {amd, 0x150e}, {amd, 0x7550}}};
  CHECK(pick_device(offered, {amd, 0x7550}) == 2u);
  CHECK(pick_device(offered, {amd, 0x1586}) == 1u);
  CHECK(pick_device(offered, {nvidia, 0x2860}) == 0u);

  const std::array<DeviceId, 1> foreign{{{0x8086, 0x7d55}}};
  CHECK_FALSE(pick_device(foreign, {amd, 0x150e}));
  CHECK_FALSE(pick_device({}, {amd, 0x150e}));
}

TEST_CASE("winml : make_runtime refuse un GPU qui n'est pas AMD, et hors de Windows") {
  ajp::gpu::vk::DeviceCaps caps;
  caps.device_name = "Intel(R) Arc(TM) Graphics";
  caps.vendor_id = 0x8086;
  auto refused = make_runtime(caps);
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::DeviceMismatch);

#ifndef _WIN32
  caps.device_name = "AMD Radeon 860M Graphics";
  caps.vendor_id = 0x1002;
  refused = make_runtime(caps);
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::LibraryUnavailable);
#endif
}
