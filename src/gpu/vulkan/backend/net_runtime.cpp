#include "gpu/vulkan/backend/net_runtime.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <fstream>
#include <unordered_set>

#include "gpu/ir/onnx_model.h"
#include "gpu/vulkan/backend/net_build.h"
#include "kernels/ref/half.h"

namespace ajp::gpu::vk::net {
namespace {

std::filesystem::path names_path(const std::filesystem::path& artifact) {
  std::filesystem::path p = artifact;
  p += ".io";
  return p;
}

std::unexpected<Refusal> cache_refusal(std::string text) {
  return std::unexpected(Refusal{.code = RefusalCode::CacheIo, .detail = std::move(text)});
}

}  // namespace

std::expected<void, Refusal> write_io_names(const std::filesystem::path& artifact,
                                            std::span<const std::string> inputs,
                                            std::span<const std::string> outputs) {
  std::string text;
  for (const std::string& name : inputs) text += std::format("in {}\n", name);
  for (const std::string& name : outputs) text += std::format("out {}\n", name);
  return write_file(names_path(artifact), std::as_bytes(std::span(text)));
}

std::expected<IoNames, Refusal> read_io_names(const std::filesystem::path& artifact) {
  std::ifstream in(names_path(artifact));
  if (!in) return cache_refusal(std::format("{} : noms des entrées absents", names_path(artifact).string()));
  IoNames names;
  std::string line;
  while (std::getline(in, line)) {
    if (line.starts_with("in ")) {
      names.inputs.push_back(line.substr(3));
    } else if (line.starts_with("out ")) {
      names.outputs.push_back(line.substr(4));
    } else if (!line.empty()) {
      return cache_refusal(std::format("{} : ligne illisible", names_path(artifact).string()));
    }
  }
  if (names.inputs.empty() || names.outputs.empty())
    return cache_refusal(std::format("{} : aucune entrée ou sortie", names_path(artifact).string()));
  return names;
}

std::expected<void, Refusal> copy_converted(const std::byte* from, TensorType from_type, std::byte* to,
                                            TensorType to_type, std::size_t count) {
  const auto size_of = [](TensorType t) -> std::size_t {
    return t == TensorType::F32 ? 4 : t == TensorType::F16 ? 2 : 1;
  };
  if (from_type == to_type) {
    std::memcpy(to, from, count * size_of(from_type));
    return {};
  }
  if (from_type == TensorType::F16 && to_type == TensorType::F32) {
    for (std::size_t i = 0; i < count; ++i) {
      std::uint16_t h = 0;
      std::memcpy(&h, from + (2 * i), 2);
      const float f = kernels::ref::half_to_float(h);
      std::memcpy(to + (4 * i), &f, 4);
    }
    return {};
  }
  if (from_type == TensorType::F32 && to_type == TensorType::F16) {
    for (std::size_t i = 0; i < count; ++i) {
      float f = 0;
      std::memcpy(&f, from + (4 * i), 4);
      const std::uint16_t h = kernels::ref::float_to_half(f);
      std::memcpy(to + (2 * i), &h, 2);
    }
    return {};
  }
  return std::unexpected(Refusal{.code = RefusalCode::FormatUnsupported,
                                 .detail = "conversion entre types entiers de tenseurs"});
}

std::expected<IoNames, Refusal> onnx_io_names(const std::filesystem::path& onnx) {
  auto bytes = read_file(onnx);
  if (!bytes) return std::unexpected(bytes.error());
  auto model = ir::onnx::read_onnx(*bytes);
  if (!model) return std::unexpected(model.error());
  // Avant l'IR 4, les poids figurent aussi parmi les entrées du graphe.
  std::unordered_set<std::string> weights;
  for (const auto& w : model->graph.initializers) weights.insert(w.name);
  IoNames names;
  for (const auto& v : model->graph.inputs) {
    if (!weights.contains(v.name)) names.inputs.push_back(v.name);
  }
  for (const auto& v : model->graph.outputs) names.outputs.push_back(v.name);
  return names;
}

std::expected<std::vector<DimOverride>, Refusal> symbolic_overrides(const std::filesystem::path& onnx,
                                                                    std::span<const InputShape> inputs) {
  auto bytes = read_file(onnx);
  if (!bytes) return std::unexpected(bytes.error());
  auto model = ir::onnx::read_onnx(*bytes);
  if (!model) return std::unexpected(model.error());
  std::vector<DimOverride> out;
  for (const InputShape& input : inputs) {
    for (const auto& v : model->graph.inputs) {
      if (v.name != input.name) continue;
      for (std::size_t i = 0; i < v.symbols.size() && i < input.dims.size(); ++i) {
        const std::string& symbol = v.symbols[i];
        if (symbol.empty() || symbol == "?") continue;  // fixe, ou dynamique sans nom
        const auto known = std::ranges::find(out, symbol, &DimOverride::first);
        if (known == out.end()) {
          out.emplace_back(symbol, input.dims[i]);
        } else if (known->second != input.dims[i]) {
          return std::unexpected(Refusal{.code = RefusalCode::InputMismatch,
                                         .detail = std::format("{} : dimension « {} » à {} et {}", input.name,
                                                               symbol, known->second, input.dims[i])});
        }
      }
    }
  }
  return out;
}

}  // namespace ajp::gpu::vk::net
