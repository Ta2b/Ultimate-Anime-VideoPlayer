#include <format>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

#include "gpu/winml/winml_detail.h"

namespace ajp::gpu::winml_rt {
namespace {

constexpr std::uint32_t kVendorAmd = 0x1002;

std::unexpected<Refusal> refuse(RefusalCode code, std::string text) {
  return std::unexpected(Refusal{.code = code, .detail = std::move(text)});
}

}  // namespace

std::filesystem::path cache_of(const std::filesystem::path& artifact) {
  std::filesystem::path p = artifact;
  p += ".mxcache";
  return p;
}

std::filesystem::path onnx_of(const std::filesystem::path& artifact) {
  std::filesystem::path p = artifact;
  p += ".onnx";
  return p;
}

std::filesystem::path npu_onnx_of(const std::filesystem::path& artifact) {
  std::filesystem::path p = artifact;
  p += ".npu.onnx";
  return p;
}

std::filesystem::path npu_cache_of(const std::filesystem::path& artifact) {
  std::filesystem::path p = artifact;
  p += ".vaicache";
  return p;
}

// Le chemin ONNX s'écrit en UTF-8, quelle que soit la page de code de Windows.
std::expected<void, Refusal> write_manifest(const std::filesystem::path& path, const Manifest& manifest) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  const std::u8string onnx = manifest.onnx.u8string();
  out << "onnx " << std::string(onnx.begin(), onnx.end()) << '\n';
  for (const auto& [symbol, value] : manifest.dims) out << "dim " << value << ' ' << symbol << '\n';
  if (manifest.io) out << "io rewritten\n";
  if (manifest.fp16) out << "fp16\n";
  if (manifest.npu) out << "npu\n";
  if (!out) return refuse(RefusalCode::CacheIo, std::format("{} : écriture", path.string()));
  return {};
}

std::expected<Manifest, Refusal> read_manifest(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return refuse(RefusalCode::CacheIo, std::format("{} : lecture", path.string()));
  Manifest manifest;
  std::string line;
  while (std::getline(in, line)) {
    if (line.starts_with("onnx ")) {
      const std::string text = line.substr(5);
      manifest.onnx = std::filesystem::path(std::u8string(text.begin(), text.end()));
    } else if (line.starts_with("dim ")) {
      std::istringstream fields(line.substr(4));
      std::int64_t value = 0;
      std::string symbol;
      fields >> value;
      std::getline(fields >> std::ws, symbol);
      if (fields.fail() && !fields.eof())
        return refuse(RefusalCode::CacheIo, std::format("{} : ligne « {} »", path.string(), line));
      if (symbol.empty())
        return refuse(RefusalCode::CacheIo, std::format("{} : ligne « {} »", path.string(), line));
      manifest.dims.emplace_back(std::move(symbol), value);
    } else if (line.starts_with("io ")) {
      manifest.io = true;
    } else if (line == "fp16") {
      manifest.fp16 = true;
    } else if (line == "npu") {
      manifest.npu = true;
    }
  }
  if (manifest.onnx.empty())
    return refuse(RefusalCode::CacheIo, std::format("{} : sans ONNX", path.string()));
  return manifest;
}

std::optional<std::size_t> pick_device(std::span<const DeviceId> offered, DeviceId chain) {
  for (std::size_t i = 0; i < offered.size(); ++i) {
    if (offered[i].vendor == chain.vendor && offered[i].device == chain.device) return i;
  }
  for (std::size_t i = 0; i < offered.size(); ++i) {
    if (offered[i].vendor == kVendorAmd) return i;
  }
  return std::nullopt;
}

}  // namespace ajp::gpu::winml_rt
