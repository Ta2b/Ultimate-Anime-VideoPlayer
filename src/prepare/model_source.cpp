// Lecture du modèle avant sa construction, propre à ajp : le fork lisait l'ONNX
// sans le comparer à une spécification.
#include "prepare/model_source.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <system_error>
#include <utility>

#include "prepare/identity.h"
#include "prepare/sha256.h"
#include "prepare/utf8.h"

namespace ajp::prepare {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

OnnxType onnx_type(gpu::TensorType type) {
  switch (type) {
    case gpu::TensorType::F32:
      return OnnxType::Float;
    case gpu::TensorType::F16:
      return OnnxType::Float16;
    case gpu::TensorType::U8:
      return OnnxType::Uint8;
    case gpu::TensorType::I8:
      return OnnxType::Int8;
  }
  return OnnxType::Undefined;
}

}  // namespace

// La taille vient du système de fichiers, jamais de tellg() : sur un dossier,
// tellg() rend une position sans rapport avec un contenu (LLONG_MAX sous ext4).
std::expected<std::vector<std::byte>, Refusal> read_onnx_bytes(const std::filesystem::path& file) {
  std::error_code error;
  const bool regular = std::filesystem::is_regular_file(file, error);
  const std::uintmax_t size = regular ? std::filesystem::file_size(file, error) : 0;
  if (!regular || error) {
    return refuse(RefusalCode::ModelUnreadable,
                  "ONNX absent ou qui n'est pas un fichier : " + utf8_path(file));
  }
  if (size == 0 || size > kMaxOnnxBytes) {
    return refuse(RefusalCode::ModelUnreadable, std::format("ONNX de {} octets, hors de 1 à {} : {}", size,
                                                            kMaxOnnxBytes, utf8_path(file)));
  }
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  std::ifstream stream(file, std::ios::binary);
  stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
  if (!stream || stream.gcount() != static_cast<std::streamsize>(size)) {
    return refuse(RefusalCode::ModelUnreadable,
                  "ONNX illisible ou tronqué pendant sa lecture : " + utf8_path(file));
  }
  return bytes;
}

std::expected<ModelSource, Refusal> read_model(const gpu::ModelSpec& model, std::string_view input_name) {
  if (!is_sha256_hex(model.sha256)) {
    return refuse(RefusalCode::FieldOutOfRange, "ModelSpec.sha256 de " + model.id + " n'est pas un SHA-256");
  }
  auto graph = read_onnx_bytes(model.onnx_path);
  if (!graph) return std::unexpected(graph.error());
  Sha256 file_hash;
  file_hash.update(*graph);
  if (file_hash.hex() != model.sha256) {
    return refuse(RefusalCode::ModelMismatch,
                  std::format("{} : SHA-256 {} au lieu de {}", utf8_path(model.onnx_path), file_hash.hex(),
                              model.sha256));
  }
  auto summary = scan_onnx(*graph);
  if (!summary) return std::unexpected(summary.error());
  for (const std::string& location : summary->external_locations) {
    if (!is_safe_location(location)) {
      return refuse(RefusalCode::ModelMismatch, "poids externe hors du dossier du modèle : " + location);
    }
  }
  auto content = onnx_source_key(*graph, model.onnx_path.parent_path(), summary->external_locations);
  if (!content) return std::unexpected(content.error());
  auto input = std::ranges::find(summary->inputs, input_name, &OnnxInput::name);
  // Un graphe aux entrées toutes statiques (GMFSS, ses tailles fixées par onnx_dims) n'a pas de
  // profil : son entrée se lit sous un autre nom, la première.
  const bool all_static =
      !summary->inputs.empty() && std::ranges::all_of(summary->inputs, &OnnxInput::is_static);
  if (input == summary->inputs.end() && all_static) input = summary->inputs.begin();
  if (input == summary->inputs.end()) {
    return refuse(RefusalCode::ModelMismatch,
                  std::format("{} : entrée « {} » absente du graphe", model.id, input_name));
  }
  for (const OnnxInput& each : summary->inputs) {
    if (auto ok = check_precision(model, each); !ok) return std::unexpected(ok.error());
  }
  OnnxInput named = *input;
  return ModelSource{.graph = std::move(*graph),
                     .summary = std::move(*summary),
                     .input = std::move(named),
                     .content_key = std::move(*content)};
}

std::expected<void, Refusal> check_precision(const gpu::ModelSpec& model, const OnnxInput& input) {
  const auto expected = static_cast<std::int32_t>(onnx_type(model.precision));
  if (input.elem_type != expected) {
    return refuse(RefusalCode::ModelMismatch, std::format("{} : entrée de type ONNX {} au lieu de {}",
                                                          model.id, input.elem_type, expected));
  }
  return {};
}

std::expected<Recipe, Refusal> recipe_for(const gpu::ModelSpec& model, const OnnxSummary& summary) {
  if (summary.grid_samples == 0) return Recipe::Upscaler;
  if (model.precision != gpu::TensorType::F16) {
    return refuse(
        RefusalCode::ModelMismatch,
        model.id + " : un graphe à GridSample se livre en FP16 avec ses îlots FP32 (convert_rife_fp16.py)");
  }
  return summary.nodes <= kTunedMaxNodes ? Recipe::Fp32IslandsTuned : Recipe::Fp32Islands;
}

std::expected<std::optional<gpu::ShapeSet>, Refusal> profile_shapes(const OnnxInput& input,
                                                                    const gpu::ShapeSet& shapes) {
  if (auto ok = check_shapes(shapes); !ok) return std::unexpected(ok.error());
  if (input.dims.size() != shapes.opt.dims.size()) {
    return refuse(RefusalCode::ModelMismatch,
                  std::format("entrée « {} » de rang {} au lieu de 4 (NCHW)", input.name, input.dims.size()));
  }
  for (std::size_t i = 0; i < input.dims.size(); ++i) {
    const std::int64_t fixed = input.dims[i];
    const bool fits = fixed < 0 || (shapes.min.dims[i] == fixed && shapes.opt.dims[i] == fixed &&
                                    shapes.max.dims[i] == fixed);
    if (!fits) {
      return refuse(RefusalCode::ModelMismatch,
                    std::format("entrée « {} » : dimension {} fixée à {} par le graphe, demandée {}",
                                input.name, i, fixed, shape_text(shapes.opt)));
    }
  }
  if (input.is_static()) return std::nullopt;
  return shapes;
}

}  // namespace ajp::prepare
