#include "app/play/player.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <system_error>
#include <utility>
#include <variant>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_timer.h>
#include <SDL3/SDL_video.h>

#include "app/play/backend_library.h"
#include "app/play/build_progress.h"
#include "app/play/chain.h"
#include "app/play/chain_stats.h"
#include "app/play/display_edid.h"
#include "app/play/engine_build.h"
#include "app/play/first_run.h"
#include "app/play/host_facts.h"
#include "app/play/input.h"
#include "app/play/model_choice.h"
#include "app/play/motion_model.h"
#include "app/play/pipeline.h"
#include "app/play/placebo_screen.h"
#include "app/play/profiles.h"
#include "app/play/scripted_input.h"
#include "app/play/session.h"
#include "app/play/ui.h"
#include "app/play/upscale_stage.h"
#include "core/clock.h"
#include "core/log.h"
#include "core/settings.h"
#include "kernels/spec/rife.h"
#include "media/demuxer.h"
#include "media/selection.h"
#include "prepare/sha256.h"
#include "vendor_hdr/sdk_store.h"
#include "vendor_hdr/true_hdr.h"

namespace ajp::app::play {
namespace {

constexpr std::uint32_t kRingSlots = 4;
// Hauteur où RIFE estime le mouvement quand la chaîne travaille plus haut (sortie 4K).
constexpr int kRifeMotionHeight = 720;
static_assert(kRingSlots <= kMaxKeptSubtitles, "les sous-titres gardés doivent couvrir tout le ring");

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

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

// La piste vidéo que la source choisira, lue d'avance : sa taille fixe le moteur et le ring.
// Le décodeur suit le backend : NVDEC en mémoire CUDA pour TensorRT ; ailleurs le décodage
// logiciel en mémoire hôte, que l'ingestion du backend Vulkan copie sur l'appareil (le décodage
// Vulkan Video sur l'appareil du rendu reste à brancher).
std::vector<media::HwApi> decoder_attempts(std::string_view backend) {
  if (backend == "tensorrt") return {media::HwApi::Cuda};
  return {media::HwApi::Software};
}

std::expected<media::TrackInfo, Refusal> probe_video(const std::filesystem::path& file) {
  auto demuxer = media::Demuxer::open(file);
  if (!demuxer) return std::unexpected(std::move(demuxer.error()));
  const media::TrackChoice choice = media::choose_tracks(demuxer->tracks(), {});
  if (!choice.video)
    return refuse(RefusalCode::TrackUnavailable, std::format("{} : aucune piste vidéo", file.string()));
  return *demuxer->track(*choice.video);
}

// Un moteur à construire, tel que la préparation le montre.
struct EngineLabel {
  std::string name;     // « SPAN », « RIFE »
  std::string model;    // nom du modèle
  std::string backend;  // « tensorrt », « vulkan »
  int width = 0;        // entrée du moteur
  int height = 0;
  double expected_s = kDefaultBuildSeconds;
};

// Une construction plus courte ne montre rien : un moteur du cache revient tout de suite.
constexpr std::int64_t kQuietBuildUs = 250'000;
// Pendant une construction, la superposition est redessinée 10 fois par seconde au plus : le GPU
// sert à mesurer les tactiques.
constexpr std::uint32_t kBuildFrameMs = 100;
// Seuils du vrai noir recalés au plus 5 fois par seconde pendant qu'on règle.
constexpr std::int64_t kBlackRefreshUs = 200'000;
// L'écran de bienvenue suit la souris : ~60 images par seconde.
constexpr std::uint32_t kWelcomeFrameMs = 16;

// Instantané demandé (--snapshot) : pris pendant la première construction de moteur qui dure jusqu'à
// at_us, sinon pendant la lecture.
struct SnapshotPlan {
  std::optional<std::filesystem::path> path;
  std::int64_t at_us = 0;
  int frames = 1;
  // Demande l'instantané à l'écran si since_us remonte à at_us ; true s'il vient d'être demandé
  // (une présentation doit suivre, même en pause).
  bool maybe(PlaceboScreen& screen, std::int64_t since_us) {
    if (!path || host_now_us() - since_us < at_us) return false;
    screen.snapshot_next(*path, frames);
    path.reset();
    return true;
  }
};

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

// Une image de la superposition avant une étape bloquante (chargement d'un moteur, RTX HDR) : la
// fenêtre dit ce qu'elle attend au lieu de rester noire.
void show_loading(PlaceboScreen& screen, PlayerUi& ui, std::string label) {
  const PrepareView view{.label = std::move(label)};
  if (auto shown = screen.show(ShowRequest{.ui = ui.preparation(host_now_us(), view)}); !shown)
    log_channel("app")->warn("préparation : {}", shown.error().detail);
}

// Ce qui survit à une réouverture de la chaîne : la fenêtre, l'interface, la bibliothèque du
// backend, les touches et l'instantané d'essai, les pistes choisies.
struct Player {
  const PlayOptions& options;
  const PlayerPaths& paths;
  PlaceboScreen& screen;
  PlayerUi& ui;
  std::shared_ptr<BackendLibrary> library;
  media::TrackInfo video;
  std::filesystem::path settings_path;
  PlaySettings from_file;  // play.toml au lancement
  PlaySettings started;    // réglages du début de la lecture (options comprises)
  SnapshotPlan snapshot;
  bool details = false;  // Tab pendant une préparation
  std::size_t next_key = 0;
  std::int64_t began_us = 0;
  std::optional<int> audio_track;
  std::optional<std::optional<int>> subtitle_track;
  ProfileState profiles;
  core::Settings engines;               // settings.toml : les moteurs choisis à l'écran de bienvenue
  std::optional<DisplayFacts> display;  // l'écran selon son EDID (OLED ou non)
};

// Fin d'une ouverture de la chaîne : la lecture est finie (rien), ou le menu demande de la rouvrir
// avec ces réglages à cette position.
struct Reopen {
  PlaySettings settings;
  std::int64_t position_us = 0;
};

// Construit (ou lit du cache) puis charge un moteur, la préparation à l'écran. Rien si
// l'utilisateur ferme la fenêtre.
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

// Une ouverture de la chaîne : moteurs, RTX HDR, threads, session et boucle de main, jusqu'à la
// fin du fichier, q, ou un réglage qui demande de la rouvrir.
std::expected<std::optional<Reopen>, Refusal> run_chain(Player& p, PlaySettings settings,
                                                        std::int64_t start_us) {
  const auto log = log_channel("app");
  std::vector<std::string> unavailable;
  settings = within_engines(settings, p.engines, unavailable);
  for (const std::string& line : unavailable) log->warn("{}", line);
  const PlayOptions options = options_for(p.options, settings);
  const media::TrackInfo& video = p.video;
  std::optional<ModelChoice> model;
  if (settings.upscale) {
    auto chosen = choose_model(p.options, p.paths.home, settings);
    if (!chosen) return std::unexpected(std::move(chosen.error()));
    model = std::move(*chosen);
  }
  const int factor = model ? kSpanFactor : 1;
  const WorkSize work = work_size(video.width, video.height, settings.work_height);
  const bool reduced = work != WorkSize{video.width, video.height};
  const std::string reduce = reduced ? std::format("{}x{} → {} {}x{}", video.width, video.height,
                                                   label(settings.downscale), work.width, work.height)
                                     : std::format("{}x{}", video.width, video.height);
  // Le modèle RIFE est cherché même sans interpolation : le menu peut l'allumer.
  std::optional<ModelChoice> rife_choice;
  if (auto found = choose_rife_model(p.options, p.paths.home, settings)) rife_choice = std::move(*found);
  const std::optional<std::filesystem::path> rife_model =
      rife_choice ? std::optional(rife_choice->path) : std::nullopt;
  // Fluidité → Moteur : le flux optique matériel remplace RIFE, sans modèle ; RIFE reprend s'il manque.
  const bool want_flow = settings.rife_factor >= 2 && settings.motion_engine == MotionEngine::OpticalFlow;
  if (settings.rife_factor >= 2 && !rife_model && !want_flow)
    return std::unexpected(choose_rife_model(p.options, p.paths.home, settings).error());
  show_loading(p.screen, p.ui, "Chargement");
  const gpu::BackendInit init{
      .render = &p.screen.device(), .cache_dir = p.paths.engines, .log = backend_log()};
  auto backend = BackendLibrary::create(p.library, init);
  if (!backend) return std::unexpected(std::move(backend.error()));
  const std::string backend_name = (*backend)->capabilities().name;
  log->info("backend {} ({}) depuis {}", (*backend)->capabilities().name,
            (*backend)->capabilities().device_name, p.library->path().string());

  std::unique_ptr<gpu::Engine> engine;
  if (model) {
    auto loaded =
        prepare_engine(p, init, **backend, model->path, gpu::Shape{.dims = {1, 3, work.height, work.width}},
                       EngineLabel{.name = "SPAN",
                                   .model = model->path.stem().string(),
                                   .width = work.width,
                                   .height = work.height});
    if (!loaded) return std::unexpected(std::move(loaded.error()));
    if (!*loaded) return std::nullopt;
    engine = std::move(**loaded);
  }
  // Au-delà de 720 lignes de travail (la 4K sans réduction), RIFE estime le mouvement sur une copie
  // réduite à 720p et le warp déforme les images de travail : la variante mouvement seulement.
  const WorkSize rife_size = !p.options.rife_image && work.height > kRifeMotionHeight
                                 ? work_size(video.width, video.height, kRifeMotionHeight)
                                 : work;
  const int flow_grid = want_flow ? open_optical_flow(*(*backend)->kernels(), rife_size.width,
                                                      rife_size.height, p.ui.controls())
                                  : 0;
  const bool use_rife = settings.rife_factor >= 2 && rife_model && flow_grid == 0;
  const kernels::RifeGeometry g = kernels::rife_geometry(rife_size.width, rife_size.height);
  const kernels::RifeGeometry image_g = kernels::rife_geometry(work.width, work.height);
  std::unique_ptr<gpu::Engine> rife;
  std::unique_ptr<gpu::Engine> rife_motion;
  // RIFE par son mouvement : une passe au milieu de la paire la sert toute quand le mouvement est
  // faible ; sans variante dérivable, RIFE image comme avant.
  if (use_rife && !p.options.rife_image) {
    auto derived = motion_model(*rife_model, p.paths.engines.parent_path() / "modeles");
    if (!derived) {
      log->warn("RIFE mouvement indisponible ({}) : RIFE image", derived.error().detail);
    } else {
      auto loaded = prepare_engine(
          p, init, **backend, *derived,
          gpu::Shape{.dims = {1, kernels::kRifeInputChannels, g.padded_height, g.padded_width}},
          EngineLabel{.name = "RIFE",
                      .model = derived->stem().string(),
                      .width = g.padded_width,
                      .height = g.padded_height});
      if (!loaded) return std::unexpected(std::move(loaded.error()));
      if (!*loaded) return std::nullopt;
      rife_motion = std::move(**loaded);
    }
  }
  if (use_rife && !rife_motion) {
    auto loaded = prepare_engine(
        p, init, **backend, *rife_model,
        gpu::Shape{.dims = {1, kernels::kRifeInputChannels, image_g.padded_height, image_g.padded_width}},
        EngineLabel{.name = "RIFE",
                    .model = rife_model->stem().string(),
                    .width = image_g.padded_width,
                    .height = image_g.padded_height});
    if (!loaded) return std::unexpected(std::move(loaded.error()));
    if (!*loaded) return std::nullopt;
    rife = std::move(**loaded);
  }
  if (rife_motion || flow_grid > 0) {
    log->info("{} : estimé en {}x{}, warp en {}x{}", rife_motion ? "RIFE mouvement" : "flux optique",
              rife_size.width, rife_size.height, work.width, work.height);
  }
  const std::string label_text = std::format(
      "{}{}{}",
      flow_grid > 0 ? std::format("flux optique ×{} · ", settings.rife_factor)
      : use_rife    ? std::format("RIFE ×{} · {} · ", settings.rife_factor, rife_model->stem().string())
                    : std::string(),
      settings.true_hdr ? "RTX HDR · " : "",
      model ? std::format("SPAN ×{} · {} · {} · {} → {}x{}", factor, model->path.stem().string(),
                          backend_name, reduce, work.width * factor, work.height * factor)
            : std::format("sans agrandissement · {}", reduce));
  log->info("{} : {}", options.file.string(), label_text);

  settings.stats.all = p.details;  // Tab pendant la préparation

  const int auto_peak = vendor_hdr::peak_from_display(content_peak_nits(p.screen.window()));
  int peak_nits = 0;
  if (settings.true_hdr) {
    show_loading(p.screen, p.ui, "Préparation de RTX HDR");
    auto opened = open_rtx_hdr(p.options.true_hdr_sdk, p.paths, settings, auto_peak, p.screen,
                               *(*backend)->kernels(), work.width * factor, work.height * factor);
    if (opened) {
      peak_nits = *opened;
    } else {
      // Le plan : TrueHDR qui échoue laisse la place à la route native, et l'écran le dit.
      log->warn("{} ; HDR natif à la place", opened.error().detail);
      p.ui.controls().notify("RTX HDR indisponible : HDR natif", host_now_us());
      settings.true_hdr = false;
      settings.hdr_native = true;
      p.screen.set_source_color(ColorInfo{});
    }
  } else {
    p.screen.set_source_color(ColorInfo{});  // celle de la source, reprise à la première image
  }

  auto live = std::make_shared<LiveSettings>();
  live->set_rife_factor(static_cast<std::uint32_t>(use_rife || flow_grid > 0 ? settings.rife_factor : 0));
  live->set_reduction(reduction_of(settings.downscale));
  live->set_keep_poses(settings.keep_poses);
  PipelineConfig config{
      .file = options.file,
      .decoder = {.attempts = decoder_attempts(backend_name), .render_device = p.screen.device().uuid},
      .audio_track = p.audio_track,
      .subtitle_track = p.subtitle_track};
  PlaceboScreen& out = p.screen;
  auto pipeline = Pipeline::open(
      config, [&](VideoStageInit stage) -> std::expected<std::unique_ptr<VideoStage>, Refusal> {
        const int width = stage.track->width;
        const int height = stage.track->height;
        const WorkSize sized = work_size(width, height, settings.work_height);
        auto slots = out.create_ring(sized.width * factor, sized.height * factor, kRingSlots);
        if (!slots) return std::unexpected(std::move(slots.error()));
        const MasterClock* clock = stage.clock;
        return UpscaleStage::start(
            std::move(stage),
            UpscaleSetup{.backend = backend->get(),
                         .engine = engine.get(),
                         .rife = rife.get(),
                         .rife_motion = rife_motion.get(),
                         .optical_flow_grid = flow_grid,
                         .rife_factor = static_cast<std::uint32_t>(std::max(settings.rife_factor, 1)),
                         .true_hdr = settings.true_hdr,
                         .live = live,
                         .clock = clock,
                         .source_width = width,
                         .source_height = height,
                         .work_width = sized.width,
                         .work_height = sized.height,
                         .rife_width = rife_motion || flow_grid > 0 ? rife_size.width : 0,
                         .rife_height = rife_motion || flow_grid > 0 ? rife_size.height : 0,
                         .slots = std::move(*slots)});
      });
  if (!pipeline) return std::unexpected(std::move(pipeline.error()));

  const bool rife_loaded = rife != nullptr || rife_motion != nullptr || flow_grid > 0;
  const ChainStats chain_stats(*live, *(*backend)->kernels(), rife_loaded,
                               flow_grid > 0 ? "flux optique" : "RIFE");
  SessionConfig session_config = (*pipeline)->session_config();
  session_config.start_us = start_us;
  session_config.stop_after_us = options.stop_after_us;
  session_config.fullscreen_after_us = options.fullscreen_after_us;
  session_config.header = {label_text};
  session_config.tracks = (*pipeline)->describe();
  p.screen.set_grade(grade_of(settings, settings.true_hdr, peak_nits, auto_peak));
  session_config.output = output_text(out, settings, peak_nits);
  session_config.display_period_us = display_period_us(out.window());
  session_config.stats = settings.stats;
  session_config.subtitle_scale = settings.subtitle_scale;
  session_config.extra = &chain_stats;
  log->info("{}", label_text);
  for (const std::string& line : session_config.tracks) log->info("{}", line);
  log->info("{}", session_config.output);
  Session session(std::move(session_config), (*pipeline)->stages(), out, (*pipeline)->clock());
  p.ui.set_tracks((*pipeline)->tracks());
  p.ui.set_settings(
      settings,
      SettingsChoices{
          .models = model ? model->available : std::vector<std::string>{},
          .default_model = model ? model->default_name : std::string(),
          .source_height = video.height,
          .rife_available = rife_model.has_value() || flow_grid > 0,
          .rife_loaded = rife_loaded,
          .rife_models = rife_choice ? rife_choice->available : std::vector<std::string>{},
          .default_rife_model = rife_choice ? rife_choice->default_name : std::string(),
          .true_hdr_available = std::filesystem::exists(p.options.true_hdr_sdk) || settings.true_hdr,
          .hdr_output = p.screen.hdr_output(),
          .auto_peak = auto_peak,
          .engines =
              engine_badges(p.paths.engines, model ? model->path : std::filesystem::path(), work,
                            rife || rife_motion ? rife_model : std::nullopt, rife_motion ? g : image_g),
          .chosen_engines = chosen_line(p.engines),
          .display_text = p.display ? describe(*p.display) : std::string("EDID illisible : OLED inconnu")});
  p.profiles.source =
      core::SourceInfo{.width = video.width, .height = video.height, .fps = fps_of(video.frame_rate)};
  if (model) p.profiles.default_model = model->default_name;
  p.ui.set_profiles(view_of(p.profiles));
  Chain chain{.session = session,
              .screen = out,
              .ui = p.ui,
              .pipeline = **pipeline,
              .live = *live,
              .running = settings,
              .profiles = p.profiles,
              .rife_loaded = rife_loaded,
              .true_hdr_open = settings.true_hdr,
              .auto_peak = auto_peak,
              .peak_nits = peak_nits,
              .display = p.display};
  apply_oled_care(chain, settings);

  std::ofstream stats;
  if (options.stats) {
    stats.open(*options.stats);
    stats << "host_us\tpts_us\tfresh\trender_us\tinterval_us\tdropped\twhy\n";
  }
  const core::KeyMap keys = core::KeyMap::defaults();
  std::optional<Refusal> failure;
  if (p.began_us == 0) p.began_us = host_now_us();
  std::vector<Input> actions;
  session.begin(host_now_us());
  // Son : le volume de la barre et le délai des réglages valent dès la première image.
  session.command(Command{p.ui.controls().volume_command()});
  if (settings.audio_delay_ms != 0)
    session.command(cmd::SetAudioDelay{.delay_us = std::int64_t{settings.audio_delay_ms} * 1000});
  std::int64_t saved_at = host_now_us();
  while (!session.finished() && !chain.reopen) {
    for (;
         p.next_key < p.options.keys.size() && host_now_us() - p.began_us >= p.options.keys[p.next_key].first;
         ++p.next_key) {
      press(p.options.keys[p.next_key].second, out.window());
    }
    const bool snapshot_now = p.snapshot.maybe(out, p.began_us);
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (p.ui.handle(event, host_now_us(), actions)) continue;  // saisie dans l'interface
      if (auto input = input_of(event, keys)) actions.push_back(*input);
    }
    (*pipeline)->pump();
    while (auto message = try_take((*pipeline)->inbox())) session.handle(std::move(*message));
    // Vrai noir : la rampe que gpu a passée par TrueHDR, puis les seuils recalés.
    if (auto ramp = chain.live.take_black_ramp()) {
      chain.truehdr_ramp = std::move(*ramp);
      chain.black_dirty = true;
    }
    if (chain.black_dirty && host_now_us() - chain.black_at_us >= kBlackRefreshUs) refresh_black(chain);
    UiFrame ui_frame = p.ui.frame(host_now_us(), session.view(), actions);
    ui_frame.changed |= snapshot_now;
    for (const Input& input : actions) apply(input, chain, host_now_us());
    actions.clear();
    // play.toml réécrit au plus une fois par seconde pendant qu'on règle.
    if (chain.dirty && host_now_us() - saved_at >= 1'000'000) {
      if (auto saved = save_settings(p.settings_path,
                                     settings_to_save(chain.running, p.from_file, p.started, p.options));
          !saved)
        log->warn("{}", saved.error().detail);
      chain.dirty = false;
      saved_at = host_now_us();
    }
    auto record = session.tick(host_now_us(), ui_frame);
    if (!record) {
      failure = std::move(record.error());
      break;
    }
    if (*record && stats.is_open()) write_record(stats, **record);
    // Rien de présenté (rien n'a changé, ou fenêtre cachée) : la FIFO n'impose pas le rythme,
    // la boucle reprend dans 1 ms pour ne manquer ni l'image suivante ni une touche.
    if (!*record) SDL_Delay(1);
  }
  if (chain.dirty) {
    if (auto saved = save_settings(p.settings_path,
                                   settings_to_save(chain.running, p.from_file, p.started, p.options));
        !saved)
      log->warn("{}", saved.error().detail);
  }
  log_summary(session);
  for (const std::string& refusal : session.refusals()) log->warn("{}", refusal);
  std::optional<Reopen> reopen;
  if (chain.reopen && !failure) {
    reopen = Reopen{.settings = *chain.reopen,
                    .position_us = std::max<std::int64_t>(session.view().position_us, 0)};
    reopen->settings.volume = p.ui.controls().volume();
    p.audio_track = (*pipeline)->tracks().audio_selected;
    p.subtitle_track = (*pipeline)->tracks().subtitle_selected;
  }
  // Ordre d'arrêt : les slots rendus, les threads arrêtés, le moteur avant son backend, le
  // backend (et ses imports du ring) avant l'écran qui possède le ring.
  session.end();
  out.flush();
  pipeline->reset();
  rife_motion.reset();
  rife.reset();
  engine.reset();
  backend->reset();
  if (failure) return std::unexpected(std::move(*failure));
  return reopen;
}

// L'écran de bienvenue jusqu'à « Commencer » ; rien si l'utilisateur quitte. Les touches d'essai
// (--keys) qui tombent pendant l'écran y sont pressées, comptées depuis son ouverture.
std::expected<std::optional<FirstRunView>, Refusal> welcome(PlaceboScreen& screen, PlayerUi& ui,
                                                            FirstRunView view, const PlayOptions& options,
                                                            std::size_t& next_key, SnapshotPlan& snapshot) {
  const core::KeyMap keys = core::KeyMap::defaults();
  const std::int64_t started = host_now_us();
  std::vector<Input> actions;
  while (!view.done) {
    for (; next_key < options.keys.size() && host_now_us() - started >= options.keys[next_key].first;
         ++next_key)
      press(options.keys[next_key].second, screen.window());
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (ui.handle(event, host_now_us(), actions)) continue;
      if (auto input = input_of(event, keys)) actions.push_back(*input);
    }
    for (const Input& input : actions) {
      if (const auto* command = std::get_if<Command>(&input)) {
        if (std::holds_alternative<cmd::Quit>(*command)) return std::nullopt;
      } else if (std::holds_alternative<ToggleFullscreen>(input)) {
        screen.set_fullscreen(!screen.fullscreen());
      } else if (std::holds_alternative<LeaveFullscreen>(input)) {
        screen.set_fullscreen(false);
      }
    }
    actions.clear();
    snapshot.maybe(screen, started);
    if (auto shown = screen.show(ShowRequest{.ui = ui.welcome(host_now_us(), view)}); !shown)
      return std::unexpected(std::move(shown.error()));
    SDL_Delay(kWelcomeFrameMs);
  }
  return view;
}

}  // namespace

std::expected<void, Refusal> play(const PlayOptions& options, const PlayerPaths& paths) {
  const auto log = log_channel("app");
  auto video = probe_video(options.file);
  if (!video) return std::unexpected(std::move(video.error()));
  const std::filesystem::path settings_path = paths.engines.parent_path() / "play.toml";
  auto saved = load_settings(settings_path);
  if (!saved) {
    log->warn("{} ; réglages par défaut", saved.error().detail);
    saved = PlaySettings{};
  }
  // Le profil de démarrage fixe la chaîne ; les options données la remplacent pour cette lecture.
  ProfileState profiles{
      .user_path = settings_path.parent_path() / "user.toml",
      .source = {.width = video->width, .height = video->height, .fps = fps_of(video->frame_rate)}};
  if (auto loaded = load_profiles(profiles.user_path)) {
    profiles.data = std::move(*loaded);
  } else {
    log->warn("{} ; profils de base", loaded.error().detail);
    profiles.data = Profiles{.file = *core::parse_profile(base_profiles_toml())};
  }
  profiles.active = profiles.data.file.start_profile;
  // Premier lancement avec profils : la chaîne de play.toml (celle d'avant les profils) devient
  // celle du profil de démarrage, au lieu d'être remplacée par la base.
  std::error_code missing;
  if (!std::filesystem::exists(profiles.user_path, missing) &&
      std::filesystem::exists(settings_path, missing)) {
    profiles.default_model = std::string(kSpanModelId);
    remember_chain(profiles, *saved);
    log->info("chaîne de play.toml reprise dans le profil {}", profiles.active);
  }
  auto screen =
      PlaceboScreen::open(ScreenOptions{.title = std::format("ajp — {}", options.file.filename().string()),
                                        .width = options.window_width,
                                        .height = options.window_height,
                                        .fullscreen = options.fullscreen,
                                        .smooth_motion = saved->smooth_motion,
                                        .allow_software = options.software_gpu});
  if (!screen) return std::unexpected(std::move(screen.error()));
  auto ui = PlayerUi::create((*screen)->window(), options.file.filename().string());
  if (!ui) return std::unexpected(std::move(ui.error()));
  // L'interface rend ses textures à l'écran avant qu'ils ne partent, quelle que soit la sortie.
  class Release {
   public:
    Release(PlayerUi& ui, PlaceboScreen& screen) : ui_(ui), screen_(screen) {}
    Release(const Release&) = delete;
    Release& operator=(const Release&) = delete;
    Release(Release&&) = delete;
    Release& operator=(Release&&) = delete;
    ~Release() { ui_.release(screen_); }

   private:
    PlayerUi& ui_;
    PlaceboScreen& screen_;
  } const release(**ui, **screen);
  SnapshotPlan snapshot{
      .path = options.snapshot, .at_us = options.snapshot_at_us, .frames = options.snapshot_frames};
  std::size_t next_key = 0;

  // Les moteurs : settings.toml, choisis à l'écran de bienvenue au premier lancement.
  const std::filesystem::path engines_path = settings_path.parent_path() / "settings.toml";
  auto engines = core::load_settings(engines_path);
  if (!engines) {
    log->warn("{} ; écran de bienvenue", engines.error().detail);
    engines = core::Settings{};
  }
  if (!engines->first_run_done || options.first_run) {
    const GpuFacts gpu = gpu_facts(**screen);
    ProfilesView choice = view_of(profiles);
    if (!engines->first_run_done) choice.start = start_profile_for(gpu);
    log->info("bienvenue : {}", gpu_line(gpu));
    auto chosen = welcome(**screen, **ui,
                          first_run_view(gpu, engines_here(options, paths), std::move(choice), *engines),
                          options, next_key, snapshot);
    if (!chosen) return std::unexpected(std::move(chosen.error()));
    if (!*chosen) return {};
    engines = settings_after(**chosen, *engines);
    if (auto written = core::save_settings(engines_path, *engines); !written)
      log->warn("{}", written.error().detail);
    const std::array<core::UserEdit, 1> start{core::edit::StartProfile{.slot = (*chosen)->profiles.start}};
    if (auto edited = edit_profiles(profiles.user_path, profiles.data, start)) {
      profiles.data = std::move(*edited);
    } else {
      log->warn("user.toml : {}", edited.error().detail);
    }
    profiles.active = profiles.data.file.start_profile;
    log->info("moteurs choisis : {} ; profil de démarrage {}", chosen_line(*engines), profiles.active);
  }

  PlaySettings from_profile = *saved;
  if (const auto& start = profiles.data.file.profiles[static_cast<std::size_t>(profiles.active - 1)]) {
    std::vector<std::string> notes;
    from_profile = apply_profile(*start, profiles.source, *saved, notes);
    for (const std::string& line : notes) log->warn("profil {} : {}", profiles.active, line);
    log->info("profil {} « {} »", profiles.active, start->name);
  }
  PlaySettings settings = settings_for(from_profile, options);
  if (auto valid = validate(settings); !valid) return std::unexpected(std::move(valid.error()));
  static_cast<void>((*ui)->controls().set_volume(settings.volume));
  show_loading(**screen, **ui, "Chargement");
  const gpu::BackendInit probe{
      .render = &(*screen)->device(), .cache_dir = paths.engines, .log = backend_log()};
  auto library = BackendLibrary::choose(backends_within(paths.backends, *engines), probe);
  if (!library) return std::unexpected(std::move(library.error()));

  Player player{.options = options,
                .paths = paths,
                .screen = **screen,
                .ui = **ui,
                .library = *library,
                .video = *video,
                .settings_path = settings_path,
                .from_file = *saved,
                .started = settings,
                .snapshot = std::move(snapshot),
                .details = settings.stats.all,
                .next_key = next_key,
                .profiles = std::move(profiles),
                .engines = std::move(*engines),
                .display = read_display(display_name((*screen)->window()))};
  if (player.display) {
    log->info("écran {} : {}", player.display->connector, describe(*player.display));
  } else {
    log->info("écran : EDID illisible, OLED inconnu");
  }
  std::int64_t start_us = options.start_us;
  std::expected<void, Refusal> result;
  while (true) {
    auto outcome = run_chain(player, settings, start_us);
    if (!outcome) {
      result = std::unexpected(std::move(outcome.error()));
      break;
    }
    if (!*outcome) break;
    settings = (*outcome)->settings;
    start_us = (*outcome)->position_us;
    player.details = settings.stats.all;
    if (auto written = save_settings(settings_path,
                                     settings_to_save(settings, player.from_file, player.started, options));
        !written)
      log->warn("{}", written.error().detail);
    log->info("réouverture à {}", format_position(start_us));
  }
  return result;
}

}  // namespace ajp::app::play
