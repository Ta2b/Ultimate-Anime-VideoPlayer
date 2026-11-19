#include "app/play/gmfss_model.h"

#include <algorithm>
#include <format>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "app/play/derived_model.h"
#include "core/utf8_path.h"
#include "prepare/model_source.h"
#include "prepare/onnx_dims.h"
#include "prepare/onnx_split.h"

namespace kirari::app::play {
namespace {

// Les dimensions nommées des deux graphes pour des plans remplis de padded_width × padded_height (multiples
// de 64) : « H », « W », puis « H/2 » à « W/8 ».
std::vector<prepare::NamedDim> gmfss_dims(int padded_width, int padded_height) {
  std::vector<prepare::NamedDim> dims{{"H", padded_height}, {"W", padded_width}};
  for (const int scale : {2, 4, 8}) {
    dims.push_back({std::format("H/{}", scale), padded_height / scale});
    dims.push_back({std::format("W/{}", scale), padded_width / scale});
  }
  return dims;
}

}  // namespace

std::vector<std::filesystem::path> gmfss_places(std::span<const std::filesystem::path> dirs) {
  std::vector<std::filesystem::path> places;
  const auto add = [&places](const std::filesystem::path& dir) {
    if (!dir.empty() && std::ranges::find(places, dir) == places.end()) places.push_back(dir);
  };
  for (const std::filesystem::path& dir : dirs) {
    add(dir);
    // « ~/kirari/models/rife/ » comme « ~/kirari/models/rife » : le voisin de rife.
    const std::filesystem::path named = dir.has_filename() ? dir : dir.parent_path();
    if (named.has_relative_path()) add(named.parent_path() / "gmfss");
  }
  return places;
}

std::optional<GmfssModels> find_gmfss(std::span<const std::filesystem::path> dirs) {
  std::error_code error;
  for (const std::filesystem::path& dir : gmfss_places(dirs)) {
    GmfssModels found{.pair = dir / kGmfssPairFile, .fusion = dir / kGmfssFusionFile};
    if (std::filesystem::is_regular_file(found.pair, error) &&
        std::filesystem::is_regular_file(found.fusion, error)) {
      if (std::filesystem::is_regular_file(dir / kGmfssHalfPairFile, error))
        found.half_pair = dir / kGmfssHalfPairFile;
      return found;
    }
  }
  return std::nullopt;
}

GmfssModels gmfss_chosen(const GmfssModels& found, bool half) {
  return {.pair = half && !found.half_pair.empty() ? found.half_pair : found.pair, .fusion = found.fusion};
}

std::expected<void, Refusal> gmfss_pair_current(const std::filesystem::path& pair) {
  auto bytes = prepare::read_onnx_bytes(pair);
  if (!bytes) return std::unexpected(std::move(bytes.error()));
  auto graph = prepare::read_onnx_graph(*bytes);
  if (!graph) return std::unexpected(std::move(graph.error()));
  for (const std::string_view name : kGmfssPairLevel1) {
    if (std::ranges::find(graph->outputs, name) == graph->outputs.end())
      return refuse(RefusalCode::GraphInvalid,
                    std::format("{} : export d'avant le format 2, sans la sortie « {} » (le réexporter par "
                                "tools/gmfss/export_gmfss.py)",
                                core::to_utf8(pair.filename()), name));
  }
  return {};
}

std::expected<GmfssModels, Refusal> gmfss_usable(const GmfssModels& found, bool half, std::string& dropped) {
  dropped.clear();
  GmfssModels chosen = gmfss_chosen(found, half);
  auto current = gmfss_pair_current(chosen.pair);
  if (current) return chosen;
  // L'autre paire du dossier : celle au flot entier pour une paire au demi-flot périmée, et l'inverse.
  const std::filesystem::path& other = chosen.pair == found.pair ? found.half_pair : found.pair;
  if (other.empty() || !gmfss_pair_current(other)) return std::unexpected(std::move(current.error()));
  dropped = std::move(current.error().detail);
  chosen.pair = other;
  return chosen;
}

std::expected<GmfssModels, Refusal> static_gmfss(const GmfssModels& models, int padded_width,
                                                 int padded_height, const std::filesystem::path& dir) {
  const std::vector<prepare::NamedDim> dims = gmfss_dims(padded_width, padded_height);
  const auto fix = [&](std::span<const std::byte> model) { return prepare::with_fixed_dims(model, dims); };
  const std::string size = std::format("{}x{}", padded_width, padded_height);
  GmfssModels fixed;
  for (const auto& [from, into] :
       {std::pair{&models.pair, &fixed.pair}, std::pair{&models.fusion, &fixed.fusion}}) {
    auto derived = derived_model(*from, size, dir, fix);
    if (!derived) return std::unexpected(std::move(derived.error()));
    // Une taille par définition de vidéo lue : les anciennes s'en vont (A09-11).
    purge_sized_derivatives(dir, core::to_utf8(from->stem()) + "-", *derived);
    *into = std::move(*derived);
  }
  return fixed;
}

}  // namespace kirari::app::play
