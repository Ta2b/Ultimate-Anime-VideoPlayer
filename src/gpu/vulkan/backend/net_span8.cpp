#include "gpu/vulkan/backend/net_span8.h"

#include <cstddef>
#include <format>
#include <fstream>
#include <system_error>
#include <vector>

#include "core/utf8_path.h"
#include "gpu/ir/import.h"

namespace ajp::gpu::vk::net {

bool is_span8(const std::filesystem::path& path) {
  return path.extension() == ".span8";
}

std::expected<ir::Graph, Refusal> span8_graph(const span_fp8::Package& package,
                                              const span_fp8::ModelOptions& options) {
  auto model = span_fp8::package_model(package, options);
  if (!model) return std::unexpected(std::move(model.error()));
  return ir::import_model(*model);
}

std::expected<ir::Graph, Refusal> load_network_graph(const std::filesystem::path& path) {
  if (!is_span8(path)) return ir::load_onnx_file(path);
  // Un fichier ordinaire, de taille plafonnée (span_fp8::kMaxPackageBytes) : un dossier ou un fichier énorme
  // n'est pas lu.
  std::error_code error;
  const std::uintmax_t size =
      std::filesystem::is_regular_file(path, error) ? std::filesystem::file_size(path, error) : 0;
  const auto unreadable = [&](std::string_view why) {
    return std::unexpected(Refusal{RefusalCode::FileUnreadable,
                                   std::format("paquet SPAN {} : {}", core::to_utf8(path.filename()), why)});
  };
  if (error || size == 0) return unreadable("illisible");
  if (size > span_fp8::kMaxPackageBytes) return unreadable("trop gros pour un paquet");
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  std::ifstream file(path, std::ios::binary);
  file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
  if (!file || file.gcount() != static_cast<std::streamsize>(size)) return unreadable("tronqué");
  auto package = span_fp8::parse(bytes);
  if (!package) return std::unexpected(std::move(package.error()));
  return span8_graph(*package);
}

}  // namespace ajp::gpu::vk::net
