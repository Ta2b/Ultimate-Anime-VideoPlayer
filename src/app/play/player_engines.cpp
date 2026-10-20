#include "app/play/player_engines.h"

#include <format>
#include <utility>
#include <variant>
#include <vector>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_timer.h>

#include "app/play/controls.h"
#include "app/play/engine_build.h"
#include "app/play/gmfss_model.h"
#include "app/play/input.h"
#include "core/log.h"
#include "kernels/spec/distil_drba.h"
#include "kernels/spec/gmfss.h"
#include "kernels/spec/rife.h"
#include "prepare/sha256.h"

namespace ajp::app::play {
namespace {

// Une construction plus courte ne montre rien : un moteur du cache revient tout de suite.
constexpr std::int64_t kQuietBuildUs = 250'000;
// Pendant une construction, la superposition est redessinée 10 fois par seconde au plus : le GPU
// sert à mesurer les tactiques.
constexpr std::uint32_t kBuildFrameMs = 100;

// Tours d'attente de main pendant la construction d'un moteur : la fenêtre reste vivante et la
// superposition montre la préparation. Rien si l'utilisateur ferme la fenêtre ou quitte (q) ; Tab
// montre les détails (details, repris ensuite par les statistiques de la lecture).
std::expected<std::optional<gpu::Artifact>, Refusal> wait_engine(EngineBuild& build, PlaceboScreen& screen,
                                                                 PlayerUi& ui, const EngineLabel& engine,
                                                                 bool& details, SnapshotPlan& snapshot) {
  const core::KeyMap keys = core::KeyMap::defaults();
  const std::int64_t started = host_now_us();
  BuildProgress progress(engine.expected_s, started);
  std::vector<Input> actions;
  while (true) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (ui.handle(event, host_now_us(), actions)) continue;
      if (auto input = input_of(event, keys)) actions.push_back(*input);
    }
    for (const Input& input : actions) {
      if (const auto* command = std::get_if<Command>(&input)) {
        if (std::holds_alternative<cmd::Quit>(*command)) return std::nullopt;
        if (std::holds_alternative<cmd::ToggleOsd>(*command)) details = !details;
      } else if (std::holds_alternative<ToggleFullscreen>(input)) {
        screen.set_fullscreen(!screen.fullscreen());
      } else if (std::holds_alternative<LeaveFullscreen>(input)) {
        screen.set_fullscreen(false);
      }
    }
    actions.clear();
    if (auto result = build.result()) {
      if (!*result) return std::unexpected(std::move(result->error()));
      log_channel("app")->info("moteur {} prêt en {:.1f} s (construction prévue {:.0f} s)", engine.name,
                               progress.elapsed_seconds(host_now_us()), engine.expected_s);
      return std::optional(std::move(**result));
    }
    const EngineBuild::Status status = build.status();
    progress.marker(status.fraction);
    const std::int64_t now = host_now_us();
    if (now - started >= kQuietBuildUs) {
      snapshot.maybe(screen, started);
      const auto seconds = [](double s) { return format_clock(static_cast<std::int64_t>(s * 1e6), 0); };
      const PrepareView view{
          .label = std::format("Préparation du moteur {}", engine.name),
          .fraction = progress.fraction(now),
          .details = {status.stage,
                      std::format("écoulé {} · prévu ~{}", seconds(progress.elapsed_seconds(now)),
                                  seconds(progress.expected_seconds())),
                      std::format("{} · entrée {}x{} · {}", engine.model, engine.width, engine.height,
                                  engine.backend)},
          .show_details = details};
      auto shown = screen.show(ShowRequest{.ui = ui.preparation(now, view)});
      if (!shown) return std::unexpected(std::move(shown.error()));
    }
    SDL_Delay(kBuildFrameMs);
  }
}

}  // namespace

void show_loading(PlaceboScreen& screen, PlayerUi& ui, std::string label) {
  const PrepareView view{.label = std::move(label)};
  if (auto shown = screen.show(ShowRequest{.ui = ui.preparation(host_now_us(), view)}); !shown)
    log_channel("app")->warn("préparation : {}", shown.error().detail);
}

std::expected<std::optional<std::unique_ptr<gpu::Engine>>, Refusal> prepare_engine(
    Player& p, const gpu::BackendInit& init, gpu::Backend& backend, const std::filesystem::path& onnx,
    const gpu::Shape& shape, EngineLabel label) {
  auto sha256 = prepare::sha256_file(onnx);
  if (!sha256) return std::unexpected(std::move(sha256.error()));
  auto build = EngineBuild::start(
      EngineRequest{.library = p.library,
                    .init = init,
                    .model = {.id = onnx.stem().string(), .onnx_path = onnx, .sha256 = *sha256},
                    .shapes = {.min = shape, .opt = shape, .max = shape}});
  if (!build) return std::unexpected(std::move(build.error()));
  label.backend = backend.capabilities().name;
  label.expected_s = expected_build_seconds(p.paths.engines, onnx, std::int64_t{label.width} * label.height);
  auto artifact = wait_engine(**build, p.screen, p.ui, label, p.details, p.snapshot);
  if (!artifact) return std::unexpected(std::move(artifact.error()));
  if (!*artifact) return std::nullopt;  // fenêtre fermée pendant la construction
  log_channel("app")->info("moteur {} {}", label.name, (*artifact)->path.string());
  show_loading(p.screen, p.ui, "Chargement des moteurs");
  auto loaded = backend.load(**artifact);
  if (!loaded) return std::unexpected(std::move(loaded.error()));
  return std::optional(std::move(*loaded));
}

std::expected<std::optional<GmfssEngines>, Refusal> prepare_gmfss(Player& p, const gpu::BackendInit& init,
                                                                  gpu::Backend& backend,
                                                                  std::span<const std::filesystem::path> dirs,
                                                                  GmfssVariant variant, int work_width,
                                                                  int work_height) {
  const std::optional<GmfssModels> found = find_gmfss(dirs, variant);
  if (!found) {
    std::string tried;
    for (const std::filesystem::path& dir : dirs)
      tried += std::format("{}{}", tried.empty() ? "" : ", ", dir.string());
    return std::unexpected(
        Refusal{RefusalCode::FieldMissing,
                std::format("GMFSS : {} et {} introuvables ({})", gmfss_files(variant).pair,
                            gmfss_files(variant).fusion, tried)});
  }
  const kernels::RifeGeometry g = kernels::rife_geometry(work_width, work_height);
  auto fixed =
      static_gmfss(*found, g.padded_width, g.padded_height, p.paths.engines.parent_path() / "modeles");
  if (!fixed) return std::unexpected(std::move(fixed.error()));
  GmfssEngines engines;
  // Les entrées des graphes : « frames » à la paire, « half0 » (la première) à la fusion.
  auto pair =
      prepare_engine(p, init, backend, fixed->pair,
                     gpu::Shape{.dims = {1, kernels::kGmfssFramePlanes, g.padded_height, g.padded_width}},
                     EngineLabel{.name = "GMFSS (paire)",
                                 .model = found->pair.stem().string(),
                                 .width = g.padded_width,
                                 .height = g.padded_height});
  if (!pair) return std::unexpected(std::move(pair.error()));
  if (!*pair) return std::nullopt;
  engines.pair = std::move(**pair);
  auto fusion = prepare_engine(p, init, backend, fixed->fusion,
                               gpu::Shape{.dims = {1, 3, g.padded_height / 2, g.padded_width / 2}},
                               EngineLabel{.name = "GMFSS (fusion)",
                                           .model = found->fusion.stem().string(),
                                           .width = g.padded_width,
                                           .height = g.padded_height});
  if (!fusion) return std::unexpected(std::move(fusion.error()));
  if (!*fusion) return std::nullopt;
  engines.fusion = std::move(**fusion);
  return std::optional(std::move(engines));
}

std::expected<std::optional<std::unique_ptr<gpu::Engine>>, Refusal> prepare_distil_drba(
    Player& p, const gpu::BackendInit& init, gpu::Backend& backend,
    std::span<const std::filesystem::path> dirs, int work_width, int work_height) {
  const std::optional<std::filesystem::path> found = find_distil_drba(dirs);
  if (!found) {
    std::string tried;
    for (const std::filesystem::path& dir : dirs)
      tried += std::format("{}{}", tried.empty() ? "" : ", ", dir.string());
    return std::unexpected(Refusal{RefusalCode::FieldMissing,
                                   std::format("DistilDRBA : {} introuvable ({})", kDistilDrbaFile, tried)});
  }
  const kernels::RifeGeometry g = kernels::rife_geometry(work_width, work_height);
  auto fixed =
      static_distil_drba(*found, g.padded_width, g.padded_height, p.paths.engines.parent_path() / "modeles");
  if (!fixed) return std::unexpected(std::move(fixed.error()));
  return prepare_engine(
      p, init, backend, *fixed,
      gpu::Shape{.dims = {1, kernels::kDistilDrbaInputPlanes, g.padded_height, g.padded_width}},
      EngineLabel{.name = "DistilDRBA",
                  .model = found->stem().string(),
                  .width = g.padded_width,
                  .height = g.padded_height});
}

}  // namespace ajp::app::play
