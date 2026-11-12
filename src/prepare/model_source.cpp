// Lecture du modèle avant sa construction, propre à ajp : le fork lisait l'ONNX
// sans le comparer à une spécification.
#include "prepare/model_source.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <system_error>
#include <utility>

#include "prepare/identity.h"
#include "prepare/sha256.h"
#include "prepare/utf8.h"

namespace ajp::prepare {
namespace {

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

// Un poids externe dont le chemin, une fois les liens symboliques suivis, sort du dossier du modèle.
// Une résolution impossible n'est pas un refus : le fichier manquant échoue plus loin, à la lecture.
// Un modèle nommé sans dossier (« --model=modele.onnx ») a un parent vide : c'est le dossier courant,
// pas une racine vide que weakly_canonical laisse vide et à laquelle tout chemin ressemble (A04-05).
bool escapes_folder(const std::filesystem::path& model_folder, const std::string& location) {
  const std::filesystem::path folder = model_folder.empty() ? std::filesystem::path(".") : model_folder;
  std::error_code root_error;
  std::error_code file_error;
  const std::filesystem::path root = std::filesystem::weakly_canonical(folder, root_error);
  const std::filesystem::path file = std::filesystem::weakly_canonical(folder / location, file_error);
  if (root_error || file_error) return false;
  return std::ranges::mismatch(root, file).in1 != root.end();
}

// Taille et date d'un fichier, ce qui suffit à reconnaître celui qu'on a déjà haché (comme
// sha256_file_known).
struct Stamp {
  std::uintmax_t size = 0;
  std::filesystem::file_time_type modified;
  bool ok = false;
  bool operator==(const Stamp&) const = default;
};

Stamp stamp_of(const std::filesystem::path& file) {
  std::error_code size_error;
  std::error_code time_error;
  Stamp stamp{.size = std::filesystem::file_size(file, size_error),
              .modified = std::filesystem::last_write_time(file, time_error)};
  stamp.ok = !size_error && !time_error;
  return stamp;
}

// La clé de contenu d'un ONNX déjà calculée dans ce processus : son SHA-256 fixe le graphe, la taille et la
// date de chaque poids externe fixent le reste. Sans elle, chaque ouverture relisait et rehachait les poids
// externes en entier, pour chaque morceau d'un RIFE coupé (C05-02).
struct KnownContent {
  std::vector<std::pair<std::filesystem::path, Stamp>> weights;
  std::string key;
};

// Les contenus déjà hachés, par SHA-256 de l'ONNX : cache du processus, sous mutex.
std::mutex g_known_mutex;                     // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::map<std::string, KnownContent> g_known;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

std::vector<std::pair<std::filesystem::path, Stamp>> weight_stamps(
    const std::filesystem::path& folder, const std::vector<std::string>& locations) {
  std::vector<std::pair<std::filesystem::path, Stamp>> stamps;
  for (const std::string& location : locations) {
    const std::filesystem::path file = folder / path_from_utf8(location);
    stamps.emplace_back(file, stamp_of(file));
  }
  return stamps;
}

bool stamps_usable(const std::vector<std::pair<std::filesystem::path, Stamp>>& stamps) {
  return std::ranges::all_of(stamps, [](const auto& each) { return each.second.ok; });
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
  const Stamp before = stamp_of(model.onnx_path);
  auto graph = read_onnx_bytes(model.onnx_path);
  if (!graph) return std::unexpected(graph.error());
  // L'appelant vient de hacher ce fichier (sha256_file_known) : tant que sa taille et sa date n'ont pas
  // bougé pendant la lecture, ce hachage vaut pour les octets lus et ils ne sont pas hachés une deuxième
  // fois.
  std::string digest;
  if (before.ok && stamp_of(model.onnx_path) == before) {
    if (auto known = sha256_file_known(model.onnx_path)) digest = std::move(*known);
  }
  if (digest.empty()) {
    Sha256 file_hash;
    file_hash.update(*graph);
    digest = file_hash.hex();
  }
  if (digest != model.sha256) {
    return refuse(RefusalCode::ModelMismatch, std::format("{} : SHA-256 {} au lieu de {}",
                                                          utf8_path(model.onnx_path), digest, model.sha256));
  }
  auto summary = scan_onnx(*graph);
  if (!summary) return std::unexpected(summary.error());
  for (const std::string& location : summary->external_locations) {
    if (!is_safe_location(location) || escapes_folder(model.onnx_path.parent_path(), location)) {
      return refuse(RefusalCode::ModelMismatch, "poids externe hors du dossier du modèle : " + location);
    }
  }
  const auto stamps = weight_stamps(model.onnx_path.parent_path(), summary->external_locations);
  std::optional<std::string> content_key;
  {
    const std::scoped_lock lock(g_known_mutex);
    if (const auto found = g_known.find(model.sha256);
        found != g_known.end() && stamps_usable(stamps) && found->second.weights == stamps)
      content_key = found->second.key;
  }
  if (!content_key) {
    auto content = onnx_source_key(*graph, model.onnx_path.parent_path(), summary->external_locations);
    if (!content) return std::unexpected(content.error());
    content_key = std::move(*content);
    // Les poids relus peuvent avoir changé pendant ce hachage : l'état d'après doit être celui d'avant.
    if (stamps_usable(stamps) &&
        weight_stamps(model.onnx_path.parent_path(), summary->external_locations) == stamps) {
      const std::scoped_lock lock(g_known_mutex);
      g_known.insert_or_assign(model.sha256, KnownContent{.weights = stamps, .key = *content_key});
    }
  }
  auto input = std::ranges::find(summary->inputs, input_name, &OnnxInput::name);
  // Un graphe aux entrées toutes statiques (DistilDRBA, ses tailles fixées par onnx_dims) n'a pas de
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
                     .content_key = std::move(*content_key)};
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
