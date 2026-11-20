#include "app/play/gmfss_model.h"

#include <algorithm>
#include <cmath>
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

int gmfss_frame_limit(double source_fps, bool half) {
  return source_fps > 0.0 ? static_cast<int>(std::ceil(source_fps * (half ? 2.5 : 2.0))) : 0;
}

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

namespace {

std::optional<GmfssModels> models_at(const std::filesystem::path& dir) {
  std::error_code error;
  GmfssModels found{.fusion = dir / kGmfssFusionFile};
  if (!std::filesystem::is_regular_file(found.fusion, error)) return std::nullopt;
  if (std::filesystem::is_regular_file(dir / kGmfssPairFile, error)) found.pair = dir / kGmfssPairFile;
  if (std::filesystem::is_regular_file(dir / kGmfssHalfPairFile, error))
    found.half_pair = dir / kGmfssHalfPairFile;
  if (found.pair.empty() && found.half_pair.empty()) return std::nullopt;
  return found;
}

}  // namespace

std::optional<GmfssModels> find_gmfss(std::span<const std::filesystem::path> dirs) {
  for (const std::filesystem::path& dir : gmfss_places(dirs)) {
    if (auto found = models_at(dir)) return found;
  }
  return std::nullopt;
}

GmfssModels gmfss_chosen(const GmfssModels& found, bool half) {
  return {.pair = !found.half_pair.empty() && (half || found.pair.empty()) ? found.half_pair : found.pair,
          .fusion = found.fusion,
          .half_pair = found.half_pair};
}

std::expected<void, Refusal> gmfss_pair_current(const std::filesystem::path& pair) {
  auto bytes = prepare::read_onnx_bytes(pair);
  if (!bytes) return std::unexpected(std::move(bytes.error()));
  auto graph = prepare::read_onnx_graph(*bytes);
  if (!graph) return std::unexpected(std::move(graph.error()));
  for (const std::string_view name : kGmfssPairLevel1) {
    if (std::ranges::find(graph->outputs, name) == graph->outputs.end())
      return refuse(RefusalCode::GraphInvalid,  // i18n-ok : refus interne, écrit au journal
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

std::expected<GmfssModels, Refusal> find_usable_gmfss(std::span<const std::filesystem::path> dirs, bool half,
                                                      std::string& dropped) {
  std::optional<Refusal> first_bad;
  for (const auto& dir : gmfss_places(dirs)) {
    const auto found = models_at(dir);
    if (!found) continue;
    auto usable = gmfss_usable(*found, half, dropped);
    if (usable) {
      if (first_bad && dropped.empty()) dropped = first_bad->detail;
      return usable;
    }
    if (!first_bad) first_bad = std::move(usable.error());
  }
  if (first_bad) return std::unexpected(std::move(*first_bad));
  return refuse(RefusalCode::ModelUnreadable,
                "GMFSS union : fusion et paire (flot entier ou demi-flot) introuvables "
                "(--gmfss-dir, dossiers de RIFE et leurs voisins gmfss)");
}

GmfssAvailability gmfss_availability(std::span<const std::filesystem::path> dirs) {
  std::string dropped;
  const auto models = find_usable_gmfss(dirs, true, dropped);
  if (!models) return {};
  return {.available = true, .half_flow = models->pair == models->half_pair};
}

GmfssAvailability GmfssMenuProbe::read(std::span<const std::filesystem::path> dirs) {
  std::vector<File> files;
  for (const auto& dir : gmfss_places(dirs)) {
    for (std::string_view name : {kGmfssPairFile, kGmfssHalfPairFile, kGmfssFusionFile}) {
      const auto path = dir / name;
      std::error_code error;
      if (!std::filesystem::is_regular_file(path, error)) continue;
      const auto size = std::filesystem::file_size(path, error);
      if (error) continue;
      const auto modified = std::filesystem::last_write_time(path, error);
      if (!error) files.push_back({.path = path, .size = size, .modified = modified});
    }
  }
  if (files != files_) {
    available_ = gmfss_availability(dirs);
    files_ = std::move(files);
  }
  return available_;
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
