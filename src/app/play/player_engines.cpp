#include "app/play/player_engines.h"

#include <format>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_timer.h>

#include "app/play/apple_model.h"
#include "app/play/controls.h"
#include "app/play/engine_build.h"
#include "app/play/gmfss_model.h"
#include "app/play/i18n.h"
#include "app/play/input.h"
#include "app/play/profiles.h"
#include "core/log.h"
#include "kernels/spec/distil_drba.h"
#include "kernels/spec/gmfss.h"
#include "kernels/spec/rife.h"
#include "prepare/drba_split.h"
#include "prepare/gmfss_split.h"
#include "prepare/rife_split.h"
#include "prepare/sha256.h"

namespace ajp::app::play {
namespace {

// Une construction plus courte ne montre rien : un moteur du cache revient tout de suite.
constexpr std::int64_t kQuietBuildUs = 250'000;
// Pendant une construction, la superposition est redessinée 10 fois par seconde au plus : le GPU
// sert à mesurer les tactiques.
constexpr std::int32_t kBuildFrameMs = 100;

// Tours d'attente de main pendant la construction d'un moteur : la fenêtre reste vivante et la
// superposition montre la préparation. Rien si l'utilisateur ferme la fenêtre ou quitte (q) ; Tab
// montre les détails (details, repris ensuite par les statistiques de la lecture).
std::expected<std::optional<gpu::Artifact>, Refusal> wait_engine(EngineBuild& build, PlaceboScreen& screen,
                                                                 PlayerUi& ui, const EngineLabel& engine,
                                                                 bool& details, SnapshotPlan& snapshot) {
  const core::KeyMap& keys = configured_keys();
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
          .label = i18n::tr_format("Preparing engine {}", engine.name),
          .fraction = progress.fraction(now),
          .details = {status.stage,
                      i18n::tr_format("elapsed {} · expected ~{}", seconds(progress.elapsed_seconds(now)),
                                      seconds(progress.expected_seconds())),
                      i18n::tr_format("{} · input {}x{} · {}", engine.model, engine.width, engine.height,
                                      engine.backend)},
          .show_details = details};
      auto shown = screen.show(ShowRequest{.ui = ui.preparation(now, view)});
      if (!shown) return std::unexpected(std::move(shown.error()));
    }
    // Une image toutes les 100 ms, plus tôt si un événement arrive.
    (void)SDL_WaitEventTimeout(nullptr, kBuildFrameMs);
  }
}

}  // namespace

gpu::BackendLog backend_log() {
  return [channel = log_channel("backend")](gpu::BackendLogLevel level, std::string_view message) {
    switch (level) {
      case gpu::BackendLogLevel::Debug:
        channel->debug("{}", message);
        break;
      case gpu::BackendLogLevel::Info:
        channel->info("{}", message);
        break;
      case gpu::BackendLogLevel::Warn:
        channel->warn("{}", message);
        break;
      case gpu::BackendLogLevel::Error:
        channel->error("{}", message);
        break;
    }
  };
}

void show_loading(PlaceboScreen& screen, PlayerUi& ui, std::string label) {
  const PrepareView view{.label = std::move(label)};
  if (auto shown = screen.show(ShowRequest{.ui = ui.preparation(host_now_us(), view)}); !shown)
    log_channel("app")->warn("préparation : {}", shown.error().detail);
}

std::expected<std::optional<std::unique_ptr<gpu::Engine>>, Refusal> prepare_engine(
    Player& p, const gpu::BackendInit& init, gpu::Backend& backend, const std::filesystem::path& onnx,
    const gpu::Shape& shape, EngineLabel label, gpu::Lane lane) {
  auto sha256 = prepare::sha256_file_known(onnx);
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
  backend.use_lane(lane);
  auto loaded = backend.load(**artifact);
  backend.use_lane(gpu::Lane::Upscale);
  if (!loaded) return std::unexpected(std::move(loaded.error()));
  return std::optional(std::move(*loaded));
}

std::expected<std::optional<std::unique_ptr<gpu::Engine>>, Refusal> prepare_rife_fused(
    Player& p, const gpu::BackendInit& init, gpu::Backend& backend, const std::filesystem::path& rife_onnx,
    const std::filesystem::path& models_dir, const kernels::RifeGeometry& geometry, bool motion) {
  auto manifest =
      prepare::derive_rife_fused(rife_onnx, models_dir, geometry.padded_height, geometry.padded_width,
                                 motion ? prepare::RifeFusedOutput::Motion : prepare::RifeFusedOutput::Image);
  if (!manifest) return std::unexpected(std::move(manifest.error()));
  return prepare_engine(
      p, init, backend, *manifest,
      gpu::Shape{.dims = {1, kernels::kRifeInputChannels, geometry.padded_height, geometry.padded_width}},
      EngineLabel{.name = i18n::tr("Fused RIFE"),
                  .model = rife_onnx.stem().string(),
                  .width = geometry.padded_width,
                  .height = geometry.padded_height},
      gpu::Lane::Interpolate);
}

namespace {

// GMFSS fusionné : la paire coupée en deux moteurs, les caractéristiques d'une image et le flot.
struct GmfssPieces {
  std::unique_ptr<gpu::Engine> features;
  std::unique_ptr<gpu::Engine> flow;
};

// Un essai des deux moteurs coupés sur des tenseurs quelconques, aux formes de la lecture, trois fois (une
// exécution à froid, une capture de graphe, un rejeu), l'entrée des caractéristiques à un décalage de plans
// de « frames » comme l'étage la donne : un moteur ou un décalage que l'appareil refuse se voit ici, avant la
// première image. Les erreurs de lancement sont rendues ; une erreur CUDA asynchrone n'est pas attendue ici.
std::expected<void, Refusal> try_gmfss_pieces(gpu::Backend& backend, gpu::Engine& features, gpu::Engine& flow,
                                              const kernels::RifeGeometry& g) {
  const auto tensor = [&](int channels, int scale) {
    return backend.alloc_tensor(
        gpu::TensorType::F16,
        gpu::Shape{.dims = {1, channels, g.padded_height / scale, g.padded_width / scale}});
  };
  auto frames = tensor(kernels::kGmfssFramePlanes, 1);
  if (!frames) return std::unexpected(std::move(frames.error()));
  std::vector<std::unique_ptr<gpu::TensorBuffer>> held;
  const auto outputs_of = [&](gpu::Engine& engine) -> std::expected<std::vector<gpu::Tensor>, Refusal> {
    std::vector<gpu::Tensor> tensors;
    for (const std::string& name : engine.output_names()) {
      int channels = 3;
      int scale = 2;  // half{k}, motion{k}_1
      if (const auto level = std::ranges::find(prepare::kGmfssFeatureOutputs, std::string_view(name));
          level != prepare::kGmfssFeatureOutputs.end()) {
        const int n = static_cast<int>(level - prepare::kGmfssFeatureOutputs.begin()) + 1;
        channels = n == 1 ? kernels::gmfss_source_channels(1) - 3 : kernels::gmfss_source_channels(n);
        scale = 1 << n;
      } else if (name.starts_with("motion") && name.size() == 9) {
        scale = 1 << (name.back() - '0');
      } else if (!name.starts_with("half")) {
        return std::unexpected(
            Refusal{RefusalCode::GraphInvalid, "GMFSS fusionné : sortie « " + name + " » inconnue"});
      }
      auto made = tensor(channels, scale);
      if (!made) return std::unexpected(std::move(made.error()));
      tensors.push_back((*made)->tensor());
      held.push_back(std::move(*made));
    }
    return tensors;
  };
  auto feature_outputs = outputs_of(features);
  if (!feature_outputs) return std::unexpected(std::move(feature_outputs.error()));
  auto flow_outputs = outputs_of(flow);
  if (!flow_outputs) return std::unexpected(std::move(flow_outputs.error()));
  gpu::Tensor second_image = (*frames)->tensor();  // les plans 3 à 5 : l'image b
  second_image.shape.dims[1] = 3;
  second_image.offset +=
      static_cast<std::uint64_t>(3 * g.padded_height * g.padded_width) * sizeof(std::uint16_t);
  for (int pass = 0; pass < 3; ++pass) {
    const std::array whole{(*frames)->tensor()};
    const std::array image{second_image};
    if (auto ok = backend.enqueue(features, image, *feature_outputs); !ok) return ok;
    if (auto ok = backend.enqueue(flow, whole, *flow_outputs); !ok) return ok;
  }
  return {};
}

// La paire de pair_onnx coupée en deux (prepare/gmfss_split.h), les deux moteurs construits, chargés et
// essayés. Rien si l'utilisateur ferme la fenêtre.
std::expected<std::optional<GmfssPieces>, Refusal> prepare_gmfss_pieces(
    Player& p, const gpu::BackendInit& init, gpu::Backend& backend, const std::filesystem::path& pair_onnx,
    const kernels::RifeGeometry& g, const std::filesystem::path& models_dir) {
  auto files = prepare::derive_gmfss_fused(pair_onnx, models_dir, g.padded_height, g.padded_width);
  if (!files) return std::unexpected(std::move(files.error()));
  GmfssPieces pieces;
  auto features = prepare_engine(p, init, backend, files->features,
                                 gpu::Shape{.dims = {1, 3, g.padded_height, g.padded_width}},
                                 EngineLabel{.name = i18n::tr("GMFSS (features)"),
                                             .model = pair_onnx.stem().string(),
                                             .width = g.padded_width,
                                             .height = g.padded_height},
                                 gpu::Lane::Interpolate);
  if (!features) return std::unexpected(std::move(features.error()));
  if (!*features) return std::nullopt;
  pieces.features = std::move(**features);
  auto flow =
      prepare_engine(p, init, backend, files->flow,
                     gpu::Shape{.dims = {1, kernels::kGmfssFramePlanes, g.padded_height, g.padded_width}},
                     EngineLabel{.name = i18n::tr("GMFSS (flow)"),
                                 .model = pair_onnx.stem().string(),
                                 .width = g.padded_width,
                                 .height = g.padded_height},
                     gpu::Lane::Interpolate);
  if (!flow) return std::unexpected(std::move(flow.error()));
  if (!*flow) return std::nullopt;
  pieces.flow = std::move(**flow);
  backend.use_lane(gpu::Lane::Interpolate);
  auto tried = try_gmfss_pieces(backend, *pieces.features, *pieces.flow, g);
  backend.use_lane(gpu::Lane::Upscale);
  if (!tried) return std::unexpected(std::move(tried.error()));
  return std::optional(std::move(pieces));
}

}  // namespace

std::expected<std::optional<GmfssEngines>, Refusal> prepare_gmfss(Player& p, const gpu::BackendInit& init,
                                                                  gpu::Backend& backend,
                                                                  std::span<const std::filesystem::path> dirs,
                                                                  GmfssVariant variant, int work_width,
                                                                  int work_height, PrecisionTier tier,
                                                                  GmfssFusedUse* fused) {
  // Le flot à demi-échelle toujours : mesuré, il vaut le flot entier en qualité (GMFlow FP32) ou le
  // dépasse, pour moitié moins de calcul ; le flot entier ne reste qu'en repli.
  // FP8 : seulement sur un moteur qui l'accélère (TensorRT, RTX 40 et après).
  if (tier == PrecisionTier::Performance && !backend.capabilities().fp8) {
    log_channel("app")->info("GMFSS Performance (FP8) : ce moteur ne le calcule pas, Équilibre à la place");
    tier = PrecisionTier::Balanced;
  }
  const std::optional<GmfssModels> found = find_gmfss(dirs, variant, true, tier);
  if (found && tier != PrecisionTier::Balanced && found->tier != tier)
    log_channel("app")->info("GMFSS {} : la paire {} manque, Équilibre à la place", label(tier),
                             gmfss_files(variant, true, tier).pair);
  if (!found) {
    std::string tried;
    for (const std::filesystem::path& dir : dirs)
      tried += std::format("{}{}", tried.empty() ? "" : ", ", dir.string());
    return std::unexpected(
        Refusal{RefusalCode::FieldMissing,
                i18n::tr_format("GMFSS: {} and {} not found ({})", gmfss_files(variant).pair,
                                gmfss_files(variant).fusion, tried)});
  }
  const kernels::RifeGeometry g = kernels::rife_geometry(work_width, work_height, found->align);
  auto fixed =
      static_gmfss(*found, g.padded_width, g.padded_height, p.paths.engines.parent_path() / "modeles");
  if (!fixed) return std::unexpected(std::move(fixed.error()));
  GmfssEngines engines{.align = found->align};
  // La paire coupée en deux d'abord : un refus de la découpe, de la construction, du chargement ou de l'essai
  // garde la paire entière.
  if (fused != nullptr) {
    auto pieces =
        prepare_gmfss_pieces(p, init, backend, found->pair, g, p.paths.engines.parent_path() / "modeles");
    if (!pieces) {
      fused->declined = pieces.error().detail;
    } else if (!*pieces) {
      return std::nullopt;
    } else {
      engines.features = std::move((**pieces).features);
      engines.pair = std::move((**pieces).flow);
      fused->used = true;
    }
  }
  // Les entrées des graphes : « frames » à la paire, « half0 » (la première) à la fusion.
  if (!engines.pair) {
    auto pair =
        prepare_engine(p, init, backend, fixed->pair,
                       gpu::Shape{.dims = {1, kernels::kGmfssFramePlanes, g.padded_height, g.padded_width}},
                       EngineLabel{.name = i18n::tr("GMFSS (pair)"),
                                   .model = found->pair.stem().string(),
                                   .width = g.padded_width,
                                   .height = g.padded_height},
                       gpu::Lane::Interpolate);
    if (!pair) return std::unexpected(std::move(pair.error()));
    if (!*pair) return std::nullopt;
    engines.pair = std::move(**pair);
  }
  auto fusion = prepare_engine(p, init, backend, fixed->fusion,
                               gpu::Shape{.dims = {1, 3, g.padded_height / 2, g.padded_width / 2}},
                               EngineLabel{.name = i18n::tr("GMFSS (fusion)"),
                                           .model = found->fusion.stem().string(),
                                           .width = g.padded_width,
                                           .height = g.padded_height},
                               gpu::Lane::Interpolate);
  if (!fusion) return std::unexpected(std::move(fusion.error()));
  if (!*fusion) return std::nullopt;
  engines.fusion = std::move(**fusion);
  return std::optional(std::move(engines));
}

std::expected<std::optional<std::unique_ptr<gpu::Engine>>, Refusal> prepare_distil_drba(
    Player& p, const gpu::BackendInit& init, gpu::Backend& backend,
    std::span<const std::filesystem::path> dirs, int work_width, int work_height, int& align,
    PrecisionTier tier, DrbaFusedUse* fused) {
  std::optional<DrbaModel> found = find_distil_drba(dirs, drba_half_flow(work_height));
  if (found && tier != PrecisionTier::Balanced) {
    if (tier == PrecisionTier::Performance && !backend.capabilities().fp8) {
      log_channel("app")->info(
          "DistilDRBA Performance (FP8) : ce moteur ne le calcule pas, Équilibre à la place");
    } else if (const std::filesystem::path variant = tier_variant(found->path, tier);
               variant != found->path) {
      log_channel("app")->info("DistilDRBA {} : {}", label(tier), variant.filename().string());
      found->path = variant;
    } else {
      log_channel("app")->info("DistilDRBA {} : {} manque, Équilibre à la place", label(tier),
                               variant.filename().string());
    }
  }
  if (!found) {
    std::string tried;
    for (const std::filesystem::path& dir : dirs)
      tried += std::format("{}{}", tried.empty() ? "" : ", ", dir.string());
    return std::unexpected(Refusal{RefusalCode::FieldMissing,
                                   i18n::tr_format("DistilDRBA: {} not found ({})", kDistilDrbaFile, tried)});
  }
  align = found->align;
  const kernels::RifeGeometry g = kernels::rife_geometry(work_width, work_height, found->align);
  const gpu::Shape shape{.dims = {1, kernels::kDistilDrbaInputPlanes, g.padded_height, g.padded_width}};
  const EngineLabel label{.name = "DistilDRBA",
                          .model = found->path.stem().string(),
                          .width = g.padded_width,
                          .height = g.padded_height};
  // Le graphe coupé d'abord : un refus de la découpe, de la construction ou du chargement garde l'entier.
  if (fused != nullptr) {
    auto manifest = prepare::derive_drba_fused(found->path, p.paths.engines.parent_path() / "modeles",
                                               g.padded_height, g.padded_width);
    if (!manifest) {
      fused->declined = manifest.error().detail;
    } else {
      auto engine = prepare_engine(p, init, backend, *manifest, shape,
                                   EngineLabel{.name = i18n::tr("Fused DistilDRBA"),
                                               .model = label.model,
                                               .width = g.padded_width,
                                               .height = g.padded_height},
                                   gpu::Lane::Interpolate);
      if (!engine) {
        fused->declined = engine.error().detail;
      } else if (!*engine) {
        return std::nullopt;  // fenêtre fermée pendant la construction
      } else {
        fused->used = true;
        return engine;
      }
    }
  }
  auto fixed = static_distil_drba(found->path, g.padded_width, g.padded_height,
                                  p.paths.engines.parent_path() / "modeles");
  if (!fixed) return std::unexpected(std::move(fixed.error()));
  return prepare_engine(p, init, backend, *fixed, shape, label, gpu::Lane::Interpolate);
}

std::expected<std::optional<std::unique_ptr<gpu::Engine>>, Refusal> prepare_apple(
    Player& p, const gpu::BackendInit& init, gpu::Backend& backend, int work_width, int work_height) {
  const std::string backend_name = backend.capabilities().name;
  if (backend_name != "coreml") {
    return std::unexpected(
        Refusal{RefusalCode::FeatureUnavailable,
                i18n::tr_format("Apple interpolation: Core ML backend only (here {})", backend_name)});
  }
  auto model = apple_model(p.paths.engines.parent_path() / "modeles");
  if (!model) return std::unexpected(std::move(model.error()));
  const kernels::RifeGeometry g = kernels::rife_geometry(work_width, work_height);
  return prepare_engine(
      p, init, backend, *model,
      gpu::Shape{.dims = {1, kernels::kRifeV2InputChannels, g.padded_height, g.padded_width}},
      EngineLabel{.name = "Apple",
                  .model = "VTFrameRateConversion",
                  .width = g.padded_width,
                  .height = g.padded_height},
      gpu::Lane::Interpolate);
}

}  // namespace ajp::app::play
