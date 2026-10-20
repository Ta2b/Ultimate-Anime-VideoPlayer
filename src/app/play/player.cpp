#include "app/play/player.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <span>
#include <string_view>
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
#include "app/play/drba_model.h"
#include "app/play/engine_build.h"
#include "app/play/engine_downloads.h"
#include "app/play/engine_install.h"
#include "app/play/first_run.h"
#include "app/play/gmfss_model.h"
#include "app/play/host_facts.h"
#include "app/play/input.h"
#include "app/play/model_choice.h"
#include "app/play/motion_model.h"
#include "app/play/pipeline.h"
#include "app/play/placebo_screen.h"
#include "app/play/player_engines.h"
#include "app/play/profiles.h"
#include "app/play/resume.h"
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
// Le décodeur suit le backend : NVDEC en mémoire CUDA pour TensorRT ; ailleurs, sous Linux, VA-API
// (le bloc vidéo d'AMD ou d'Intel, même sur le GPU intégré d'un portable) copié en mémoire hôte,
// puis le décodage logiciel. L'ingestion du backend Vulkan copie l'image hôte sur l'appareil (le
// décodage Vulkan Video sur l'appareil du rendu reste à brancher).
std::vector<media::HwApi> decoder_attempts(std::string_view backend) {
  if (backend == "tensorrt") return {media::HwApi::Cuda};
#if defined(__linux__)
  return {media::HwApi::Vaapi, media::HwApi::Software};
#else
  return {media::HwApi::Software};
#endif
}

std::expected<media::TrackInfo, Refusal> probe_video(const std::filesystem::path& file) {
  auto demuxer = media::Demuxer::open(file);
  if (!demuxer) return std::unexpected(std::move(demuxer.error()));
  const media::TrackChoice choice = media::choose_tracks(demuxer->tracks(), {});
  if (!choice.video)
    return refuse(RefusalCode::TrackUnavailable, std::format("{} : aucune piste vidéo", file.string()));
  return *demuxer->track(*choice.video);
}

// Seuils du vrai noir recalés au plus 5 fois par seconde pendant qu'on règle.
constexpr std::int64_t kBlackRefreshUs = 200'000;
// L'écran de bienvenue suit la souris : ~60 images par seconde.
constexpr std::uint32_t kWelcomeFrameMs = 16;

// Fin d'une ouverture de la chaîne : la lecture est finie (rien), ou le menu demande de la rouvrir
// avec ces réglages à cette position.
struct Reopen {
  PlaySettings settings;
  std::int64_t position_us = 0;
};

// Où chercher les graphes de GMFSS : à côté du modèle RIFE choisi, puis dans les dossiers RIFE.
std::vector<std::filesystem::path> gmfss_dirs(const std::filesystem::path& home,
                                              const std::optional<ModelChoice>& rife) {
  std::vector<std::filesystem::path> dirs;
  if (rife) dirs.push_back(rife->path.parent_path());
  for (const std::filesystem::path& candidate : default_rife_candidates(home)) {
    if (std::ranges::find(dirs, candidate.parent_path()) == dirs.end())
      dirs.push_back(candidate.parent_path());
  }
  return dirs;
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
  const std::span<const std::string_view> rife_order = rife_preference(gpu_facts(p.screen));
  std::optional<ModelChoice> rife_choice;
  if (auto found = choose_rife_model(p.options, p.paths.home, settings, rife_order))
    rife_choice = std::move(*found);
  const std::optional<std::filesystem::path> rife_model =
      rife_choice ? std::optional(rife_choice->path) : std::nullopt;
  const std::vector<std::filesystem::path> gmfss_places = gmfss_dirs(p.paths.home, rife_choice);
  const GmfssVariant gmfss_variant =
      settings.motion_engine == MotionEngine::GmfssUnion ? GmfssVariant::Union : GmfssVariant::Base;
  const bool gmfss_found = find_gmfss(gmfss_places, gmfss_variant).has_value();
  const bool want_gmfss = settings.rife_factor >= 2 && is_gmfss(settings.motion_engine);
  const bool drba_found = find_distil_drba(gmfss_places).has_value();
  const bool want_drba = settings.rife_factor >= 2 && settings.motion_engine == MotionEngine::DistilDrba;
  if (settings.rife_factor >= 2 && !rife_model && !(want_gmfss && gmfss_found) && !(want_drba && drba_found))
    return std::unexpected(choose_rife_model(p.options, p.paths.home, settings, rife_order).error());
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
  // GMFSS Fortuna à la place de RIFE, à la taille de travail ; RIFE reprend si ses graphes manquent.
  GmfssEngines gmfss;
  if (want_gmfss) {
    auto loaded = prepare_gmfss(p, init, **backend, gmfss_places, gmfss_variant, work.width, work.height);
    if (!loaded) {
      log->warn("{} ; RIFE à la place", loaded.error().detail);
      p.ui.controls().notify("GMFSS indisponible : RIFE", host_now_us());
    } else if (!*loaded) {
      return std::nullopt;
    } else {
      gmfss = std::move(**loaded);
    }
  }
  const bool use_gmfss = gmfss.pair != nullptr;
  // DistilDRBA à la place de RIFE, à la taille de travail ; RIFE reprend si son graphe manque.
  std::unique_ptr<gpu::Engine> drba;
  if (want_drba) {
    auto loaded = prepare_distil_drba(p, init, **backend, gmfss_places, work.width, work.height);
    if (!loaded) {
      log->warn("{} ; RIFE à la place", loaded.error().detail);
      p.ui.controls().notify("DistilDRBA indisponible : RIFE", host_now_us());
    } else if (!*loaded) {
      return std::nullopt;
    } else {
      drba = std::move(**loaded);
    }
  }
  const bool use_drba = drba != nullptr;
  const bool use_rife = settings.rife_factor >= 2 && rife_model && !use_gmfss && !use_drba;
  const int rife_align = use_rife ? rife_padding_multiple(*rife_model) : kernels::kRifeAlign;
  const kernels::RifeGeometry g = kernels::rife_geometry(rife_size.width, rife_size.height, rife_align);
  const kernels::RifeGeometry image_g = kernels::rife_geometry(work.width, work.height, rife_align);
  std::unique_ptr<gpu::Engine> rife;
  std::unique_ptr<gpu::Engine> rife_motion;
  int rife_planes = kernels::kRifeInputChannels;
  if (use_rife) {
    auto planes = rife_input_planes(*rife_model);
    if (!planes) return std::unexpected(std::move(planes.error()));
    rife_planes = *planes;
  }
  // RIFE par son mouvement : une passe au milieu de la paire la sert toute quand le mouvement est
  // faible ; sans variante dérivable, RIFE image comme avant.
  if (use_rife && !p.options.rife_image) {
    auto derived = motion_model(*rife_model, p.paths.engines.parent_path() / "modeles");
    if (!derived) {
      log->warn("RIFE mouvement indisponible ({}) : RIFE image", derived.error().detail);
    } else {
      auto loaded = prepare_engine(p, init, **backend, *derived,
                                   gpu::Shape{.dims = {1, rife_planes, g.padded_height, g.padded_width}},
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
    auto loaded =
        prepare_engine(p, init, **backend, *rife_model,
                       gpu::Shape{.dims = {1, rife_planes, image_g.padded_height, image_g.padded_width}},
                       EngineLabel{.name = "RIFE",
                                   .model = rife_model->stem().string(),
                                   .width = image_g.padded_width,
                                   .height = image_g.padded_height});
    if (!loaded) return std::unexpected(std::move(loaded.error()));
    if (!*loaded) return std::nullopt;
    rife = std::move(**loaded);
  }
  if (rife_motion) {
    log->info("RIFE mouvement : estimé en {}x{}, warp en {}x{}", rife_size.width, rife_size.height,
              work.width, work.height);
  }

  settings.stats.all = p.details;  // Tab pendant la préparation

  const int auto_peak = vendor_hdr::peak_from_display(content_peak_nits(p.screen.window()));
  int peak_nits = 0;
  // La sortie est HDR10 : l'écran en reçoit, et --sdr ne l'a pas coupée. Sinon (sortie SDR),
  // TrueHDR ferait du HDR que libplacebo ramènerait aussitôt en SDR ; le grade SDR prend sa place.
  const bool hdr_out = !p.options.sdr && content_peak_nits(p.screen.window()).has_value();
  if (settings.true_hdr && !hdr_out) {
    log->info("RTX HDR : sortie SDR, TrueHDR n'est pas ouvert ; grade SDR à la place");
    settings.true_hdr = false;
  }
  if (settings.true_hdr) {
    show_loading(p.screen, p.ui, "Préparation de RTX HDR");
    auto opened = open_rtx_hdr(p.options.true_hdr_sdk, p.paths, settings, auto_peak, p.screen,
                               *(*backend)->kernels(), work.width * factor, work.height * factor);
    if (opened) {
      peak_nits = *opened;
    } else {
      // Le plan : TrueHDR qui échoue laisse la place à la route native, et l'écran le dit.
      log->warn("{} ; HDR adaptatif à la place", opened.error().detail);
      p.ui.controls().notify("RTX HDR indisponible : HDR adaptatif", host_now_us());
      settings.true_hdr = false;
      settings.hdr_native = true;
      p.screen.set_source_color(ColorInfo{});
    }
  } else {
    p.screen.set_source_color(ColorInfo{});  // celle de la source, reprise à la première image
  }

  const std::string label_text = std::format(
      "{}{}{}",
      use_gmfss  ? std::format("GMFSS Fortuna{} {} · ", gmfss_variant == GmfssVariant::Union ? " Union" : "",
                               factor_text(factor_of(settings)))
      : use_drba ? std::format("DistilDRBA {} · ", factor_text(factor_of(settings)))
      : use_rife
          ? std::format("RIFE {} · {} · ", factor_text(factor_of(settings)), rife_model->stem().string())
          : std::string(),
      // Le moteur HDR vraiment pris : TrueHDR ouvert, ou le HDR adaptatif sur une sortie HDR10.
      settings.true_hdr ? "RTX HDR · " : (settings.hdr_native && hdr_out ? "HDR adaptatif · " : ""),
      model ? std::format("SPAN ×{} · {} · {} · {} → {}x{}", factor, model->path.stem().string(),
                          backend_name, reduce, work.width * factor, work.height * factor)
            : std::format("sans agrandissement · {}", reduce));
  log->info("{} : {}", options.file.string(), label_text);
  auto live = std::make_shared<LiveSettings>();
  live->set_rife_factor(use_rife || use_gmfss || use_drba ? factor_of(settings) : Factor{});
  live->set_reduction(reduction_of(settings.downscale));
  live->set_keep_poses(settings.keep_poses);
  PipelineConfig config{.file = options.file,
                        .decoder = {.attempts = decoder_attempts(backend_name),
                                    .render_device = p.screen.device().uuid,
                                    .download = backend_name != "tensorrt"},
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
        return UpscaleStage::start(std::move(stage),
                                   UpscaleSetup{.backend = backend->get(),
                                                .engine = engine.get(),
                                                .rife = rife.get(),
                                                .rife_motion = rife_motion.get(),
                                                .rife_planes = rife_planes,
                                                .rife_align = rife_align,
                                                .gmfss_pair = gmfss.pair.get(),
                                                .gmfss_fusion = gmfss.fusion.get(),
                                                .gmfss_align = gmfss.align,
                                                .distil_drba = drba.get(),
                                                .rife_factor = factor_of(settings),
                                                .true_hdr = settings.true_hdr,
                                                .live = live,
                                                .clock = clock,
                                                .source_width = width,
                                                .source_height = height,
                                                .work_width = sized.width,
                                                .work_height = sized.height,
                                                .rife_width = rife_motion ? rife_size.width : 0,
                                                .rife_height = rife_motion ? rife_size.height : 0,
                                                .warp_after_span = settings.warp_after_span,
                                                .slots = std::move(*slots)});
      });
  if (!pipeline) return std::unexpected(std::move(pipeline.error()));

  const bool rife_loaded = rife != nullptr || rife_motion != nullptr || use_gmfss || use_drba;
  const ChainStats chain_stats(*live, *(*backend)->kernels(), rife_loaded,
                               use_gmfss  ? "GMFSS"
                               : use_drba ? "DistilDRBA"
                                          : "RIFE");
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
          .source_fps = fps_of(video.frame_rate),
          .display_hz = display_period_us(out.window()) > 0
                            ? 1e6 / static_cast<double>(display_period_us(out.window()))
                            : 0.0,
          .rife_available = rife_model.has_value() || gmfss_found || drba_found,
          .gmfss_available = find_gmfss(gmfss_places).has_value(),
          .gmfss_union_available = find_gmfss(gmfss_places, GmfssVariant::Union).has_value(),
          .distil_drba_available = drba_found,
          .rife_loaded = rife_loaded,
          .rife_models = rife_choice ? rife_choice->available : std::vector<std::string>{},
          .default_rife_model = rife_choice ? rife_choice->default_name : std::string(),
          .rife_folder = gmfss_places.front().string(),
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
  if (p.resumed) {
    p.ui.controls().notify(std::format("Reprise à {}", format_clock(start_us, session.view().duration_us)),
                           host_now_us());
    p.resumed = false;
  }
  // La position retenue toutes les kResumeEveryUs, et à la fin : un arrêt brutal en perd peu.
  std::int64_t kept_us = kNoPts;
  const auto remember = [&] {
    if (p.resume_path.empty()) return;
    const ControlsView view = session.view();
    if (view.position_us == kNoPts || view.position_us == kept_us) return;
    kept_us = view.position_us;
    if (auto kept = remember_position(p.resume_path, p.options.file, view.position_us, view.duration_us);
        !kept)
      log->warn("{}", kept.error().detail);
  };
  std::int64_t remembered_at = host_now_us();
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
    if (host_now_us() - remembered_at >= kResumeEveryUs) {
      remember();
      remembered_at = host_now_us();
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
  remember();
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
                                        .hdr = !options.sdr,
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
  // Les moteurs téléchargeables de cette version (engines.lock livré avec le lecteur).
  const EngineCatalog catalog = load_catalog(paths.engine_index, paths.engines.parent_path());
  if (!catalog.problem.empty()) log->info("{}", catalog.problem);
  if (!engines->first_run_done || options.first_run) {
    const GpuFacts gpu = gpu_facts(**screen);
    ProfilesView choice = view_of(profiles);
    if (!engines->first_run_done) choice.start = start_profile_for(gpu);
    log->info("bienvenue : {}", gpu_line(gpu));
    auto chosen =
        welcome(**screen, **ui,
                first_run_view(gpu, engines_here(options, paths, catalog), std::move(choice), *engines),
                options, next_key, snapshot);
    if (!chosen) return std::unexpected(std::move(chosen.error()));
    if (!*chosen) return {};
    engines = settings_after(**chosen, *engines);
    auto downloaded =
        install_chosen(**screen, **ui, catalog, **chosen, *engines, paths.engines.parent_path() / "sdk");
    if (!downloaded) return std::unexpected(std::move(downloaded.error()));
    if (!*downloaded) return {};
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
  // Les backends téléchargés d'abord, puis ceux livrés ou de l'arbre de construction.
  std::vector<std::filesystem::path> candidates = installed_backends(catalog);
  candidates.insert(candidates.end(), paths.backends.begin(), paths.backends.end());
  auto library = BackendLibrary::choose(backends_within(candidates, *engines), probe);
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
  // settings.toml ([playback] resume_position) l'autorise, et --no-resume l'écarte pour une lecture.
  if (options.resume && player.engines.resume_position) {
    player.resume_path = settings_path.parent_path() / "resume.tsv";
    if (start_us == 0) {
      if (const auto at = resume_position(player.resume_path, options.file)) {
        start_us = *at;
        player.resumed = true;
        log->info("reprise à {}", format_position(start_us));
      }
    }
  }
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
