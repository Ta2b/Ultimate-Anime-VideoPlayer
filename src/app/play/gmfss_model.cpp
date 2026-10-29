#include "app/play/gmfss_model.h"

#include <algorithm>
#include <charconv>
#include <format>
#include <map>
#include <mutex>
#include <set>
#include <system_error>

#include "app/play/derived_model.h"
#include "core/log.h"
#include "kernels/spec/rife.h"
#include "prepare/model_source.h"
#include "prepare/onnx_metadata.h"
#include "prepare/onnx_outputs.h"

namespace ajp::app::play {
namespace {

// Les tenseurs que produit le GridSample de la fenêtre locale de GMFlow des exports d'avant la lecture par
// décalages entiers : « /flownet/GridSample_1 », « /flownet_1/GridSample_1 ». Les exports courants n'ont plus
// que « /flownet/GridSample » (la propagation du flot).
bool reads_window_by_grid_sample(const std::vector<std::string>& tensors) {
  return std::ranges::any_of(tensors, [](const std::string& name) {
    return name.starts_with("/flownet") && name.contains("/GridSample_1");
  });
}

GmfssGraphState inspect(const std::filesystem::path& file) {
  const auto value = prepare::onnx_metadata(file, kGmfssExportKey);
  if (!value) return {};  // illisible : la construction le dira
  int version = 1;
  if (*value) {
    const std::string& text = **value;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), version);
    if (error != std::errc{} || end != text.data() + text.size() || version < 1) {
      return {.state = GmfssExport::Stale, .version = 0, .reason = "version d'export illisible : " + text};
    }
  }
  if (version == kGmfssExportVersion) return {};
  if (version > kGmfssExportVersion) {
    return {.state = GmfssExport::Stale,
            .version = version,
            .reason = std::format("export de version {}, plus récent que ce lecteur ({})", version,
                                  kGmfssExportVersion)};
  }
  // Une version antérieure : lisible, sauf si la fenêtre de GMFlow y passe encore par le GridSample.
  const auto bytes = prepare::read_onnx_bytes(file);
  if (!bytes) return {};
  const auto tensors = prepare::onnx_node_outputs(*bytes);
  if (!tensors) return {};
  if (reads_window_by_grid_sample(*tensors)) {
    return {.state = GmfssExport::Stale,
            .version = version,
            .reason =
                "la fenêtre locale de GMFlow y passe par un GridSample de 159 millions de valeurs "
                "(2 x 7,2 ms par paire à 1280x768)"};
  }
  return {.state = GmfssExport::Legacy,
          .version = version,
          .reason = std::format("export de version {} (courante {}) : les 64 canaux et l'image sont "
                                "concaténés dans la paire, 0,7 ms de plus par paire à 1280x768",
                                version, kGmfssExportVersion)};
}

// Une même ligne de journal une seule fois par fichier et par état : find_gmfss est appelée à chaque
// ouverture de menu et de chaîne.
void say_once(const std::filesystem::path& file, const GmfssGraphState& state) {
  static std::mutex mutex;
  static std::set<std::string> said;
  const std::lock_guard lock(mutex);
  if (!said.insert(std::format("{}|{}", file.string(), static_cast<int>(state.state))).second) return;
  if (state.state == GmfssExport::Stale) {
    log_channel("app")->warn(
        "graphe GMFSS périmé, ré-export : {} non utilisé ({}) ; ré-exporter avec tools/gmfss/export_gmfss.py "
        "ou télécharger la release des modèles d'interpolation",
        file.string(), state.reason);
  } else {
    log_channel("app")->info("graphe GMFSS d'un export ancien, lu tel quel : {} ({}) ; ré-export conseillé",
                             file.string(), state.reason);
  }
}

}  // namespace

GmfssGraphState gmfss_graph_state(const std::filesystem::path& file) {
  // Un export sans métadonnée coûte une lecture complète du graphe : faite une fois par état du fichier
  // (taille et date).
  static std::mutex mutex;
  static std::map<std::string, GmfssGraphState> known;
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(file, error);
  const auto written = std::filesystem::last_write_time(file, error);
  if (error) return {};
  const std::string key = std::format("{}|{}|{}", file.string(), size, written.time_since_epoch().count());
  {
    const std::lock_guard lock(mutex);
    if (const auto found = known.find(key); found != known.end()) return found->second;
  }
  GmfssGraphState state = inspect(file);
  const std::lock_guard lock(mutex);
  known.emplace(key, state);
  return state;
}

std::optional<GmfssModels> find_gmfss(std::span<const std::filesystem::path> dirs, GmfssVariant variant,
                                      bool prefer_half, PrecisionTier tier) {
  std::error_code error;
  // Les essais dans l'ordre : la paire du cran demandé (demi-échelle), puis les graphes du flot à
  // demi-échelle et du flot entier, dans l'ordre que prefer_half donne.
  struct Attempt {
    bool half_flow;
    PrecisionTier tier;
  };
  std::vector<Attempt> attempts;
  if (tier != PrecisionTier::Balanced) attempts.push_back({true, tier});
  for (const bool half_flow : {prefer_half, !prefer_half})
    attempts.push_back({half_flow, PrecisionTier::Balanced});
  for (const auto& [half_flow, attempt_tier] : attempts) {
    const GmfssFiles files = gmfss_files(variant, half_flow, attempt_tier);
    for (const std::filesystem::path& dir : dirs) {
      GmfssModels found{.pair = dir / files.pair,
                        .fusion = dir / files.fusion,
                        .align = half_flow ? kGmfssHalfAlign : kernels::kRifeAlign,
                        .tier = attempt_tier};
      if (std::filesystem::is_regular_file(found.pair, error) &&
          std::filesystem::is_regular_file(found.fusion, error)) {
        const GmfssGraphState state = gmfss_graph_state(found.pair);
        if (state.state != GmfssExport::Current) say_once(found.pair, state);
        if (state.state == GmfssExport::Stale) continue;  // jamais utilisé : l'essai suivant
        return found;
      }
    }
  }
  return std::nullopt;
}

bool gmfss_tier_available(std::span<const std::filesystem::path> dirs, GmfssVariant variant,
                          PrecisionTier tier) {
  const std::optional<GmfssModels> found = find_gmfss(dirs, variant, true, tier);
  return found && found->tier == tier;
}

TierSupport gmfss_tiers(std::span<const std::filesystem::path> dirs, GmfssVariant variant, bool fp8_engine) {
  // Le cran Performance (FP8) de GMFSS n'est plus proposé : son image se dégrade dans les gros mouvements
  // (retour de Tatoub sur la 0.7.0-rc.1). Les graphes FP8 restent au catalogue, sans être offerts ni choisis.
  static_cast<void>(fp8_engine);
  return TierSupport{.performance = false,
                     .quality = gmfss_tier_available(dirs, variant, PrecisionTier::Quality)};
}

std::vector<prepare::NamedDim> gmfss_dims(int padded_width, int padded_height) {
  std::vector<prepare::NamedDim> dims{{"H", padded_height}, {"W", padded_width}};
  for (const int scale : {2, 4, 8}) {
    dims.push_back({std::format("H/{}", scale), padded_height / scale});
    dims.push_back({std::format("W/{}", scale), padded_width / scale});
  }
  return dims;
}

std::expected<GmfssModels, Refusal> static_gmfss(const GmfssModels& models, int padded_width,
                                                 int padded_height, const std::filesystem::path& dir) {
  const std::vector<prepare::NamedDim> dims = gmfss_dims(padded_width, padded_height);
  const auto fix = [&](std::span<const std::byte> model) { return prepare::with_fixed_dims(model, dims); };
  const std::string size = std::format("{}x{}", padded_width, padded_height);
  auto pair = derived_model(models.pair, size, dir, fix);
  if (!pair) return std::unexpected(std::move(pair.error()));
  auto fusion = derived_model(models.fusion, size, dir, fix);
  if (!fusion) return std::unexpected(std::move(fusion.error()));
  return GmfssModels{.pair = std::move(*pair), .fusion = std::move(*fusion), .align = models.align};
}

}  // namespace ajp::app::play
