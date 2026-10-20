#include "app/play/player.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
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
#include "app/play/chain_setup.h"
#include "app/play/chain_stats.h"
#include "app/play/display_edid.h"
#include "app/play/engine_build.h"
#include "app/play/engine_install.h"
#include "app/play/engine_screen.h"
#include "app/play/engine_switch.h"
#include "app/play/export_run.h"
#include "app/play/first_run.h"
#include "app/play/gmfss_model.h"
#include "app/play/host_facts.h"
#include "app/play/input.h"
#include "app/play/model_choice.h"
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
#include "media/demuxer.h"
#include "media/selection.h"
#include "prepare/sha256.h"
#include "vendor_hdr/sdk_store.h"
#include "vendor_hdr/true_hdr.h"

namespace ajp::app::play {
namespace {

constexpr std::uint32_t kRingSlots = 4;
static_assert(kRingSlots <= kMaxKeptSubtitles, "les sous-titres gardés doivent couvrir tout le ring");

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

// La piste vidéo que la source choisira, lue d'avance : sa taille fixe le moteur et le ring.
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

// Fin d'une ouverture de la chaîne : la lecture est finie (rien), ou le menu demande de la rouvrir
// avec ces réglages à cette position.
struct Reopen {
  PlaySettings settings;
  std::int64_t position_us = 0;
  std::optional<EngineChange> engines;  // onglet Moteurs : à faire avant de rouvrir
  std::optional<ExportJob> export_job;  // onglet Export : à faire avant de rouvrir
};

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
  auto prepared = prepare_chain(p, settings);
  if (!prepared) return std::unexpected(std::move(prepared.error()));
  if (!*prepared) return std::nullopt;
  ChainEngines& c = **prepared;
  const int factor = c.factor;
  const WorkSize work = c.work;
  const bool rife_loaded = c.interpolates();

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
                               *c.backend->kernels(), work.width * factor, work.height * factor);
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
      "{}{}{}", c.motion_label(settings),
      // Le moteur HDR vraiment pris : TrueHDR ouvert, ou le HDR adaptatif sur une sortie HDR10.
      settings.true_hdr ? "RTX HDR · " : (settings.hdr_native && hdr_out ? "HDR adaptatif · " : ""),
      c.span_label());
  log->info("{} : {}", options.file.string(), label_text);
  auto live = std::make_shared<LiveSettings>();
  live->set_rife_factor(rife_loaded ? factor_of(settings) : Factor{});
  live->set_reduction(reduction_of(settings.downscale));
  live->set_keep_poses(settings.keep_poses);
  PipelineConfig config{.file = options.file,
                        .decoder = {.attempts = decoder_attempts(c.backend_name),
                                    .render_device = p.screen.device().uuid,
                                    .download = c.backend_name != "tensorrt"},
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
                                   c.upscale(settings, live, clock, width, height, std::move(*slots)));
      });
  if (!pipeline) return std::unexpected(std::move(pipeline.error()));

  const ChainStats chain_stats(*live, *c.backend->kernels(), rife_loaded, std::string(c.motion_name()));
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
          .models = c.model ? c.model->available : std::vector<std::string>{},
          .default_model = c.model ? c.model->default_name : std::string(),
          .source_width = video.width,
          .source_height = video.height,
          .source_fps = fps_of(video.frame_rate),
          .source_rate = video.frame_rate,
          .display_hz = display_period_us(out.window()) > 0
                            ? 1e6 / static_cast<double>(display_period_us(out.window()))
                            : 0.0,
          .rife_available = c.rife_model.has_value() || c.gmfss_found || c.drba_found,
          .gmfss_available = find_gmfss(c.gmfss_places).has_value(),
          .gmfss_union_available = find_gmfss(c.gmfss_places, GmfssVariant::Union).has_value(),
          .distil_drba_available = c.drba_found,
          .rife_loaded = rife_loaded,
          .rife_models = c.rife_choice ? c.rife_choice->available : std::vector<std::string>{},
          .default_rife_model = c.rife_choice ? c.rife_choice->default_name : std::string(),
          .rife_folder = c.gmfss_places.front().string(),
          .true_hdr_available = std::filesystem::exists(p.options.true_hdr_sdk) || settings.true_hdr,
          .hdr_output = p.screen.hdr_output(),
          .auto_peak = auto_peak,
          .engines = engine_badges(p.paths.engines, c.model ? c.model->path : std::filesystem::path(), work,
                                   c.rife || c.rife_motion ? c.rife_model : std::nullopt,
                                   c.rife_motion ? c.geometry : c.image_geometry),
          .chosen_engines = chosen_line(p.engines),
          .switchable = installed_switchable(p.catalog, p.paths),
          .engine_in_use = engine_of_library(p.library->path()),
          .engine_note = fallback_note(p.engines, p.library->path()),
          .display_text = p.display ? describe(*p.display) : std::string("EDID illisible : OLED inconnu"),
          .source_file = p.options.file});
  p.profiles.source =
      core::SourceInfo{.width = video.width, .height = video.height, .fps = fps_of(video.frame_rate)};
  if (c.model) p.profiles.default_model = c.model->default_name;
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
  if (!p.notice.empty()) {
    p.ui.controls().notify(p.notice, host_now_us());
    p.notice.clear();
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
                    .position_us = std::max<std::int64_t>(session.view().position_us, 0),
                    .engines = chain.engines,
                    .export_job = chain.export_job};
    reopen->settings.volume = p.ui.controls().volume();
    p.audio_track = (*pipeline)->tracks().audio_selected;
    p.subtitle_track = (*pipeline)->tracks().subtitle_selected;
  }
  // Ordre d'arrêt : les slots rendus, les threads arrêtés, le moteur avant son backend, le
  // backend (et ses imports du ring) avant l'écran qui possède le ring.
  session.end();
  out.flush();
  pipeline->reset();
  prepared->reset();
  if (failure) return std::unexpected(std::move(*failure));
  return reopen;
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
  const EngineScreen engine_screen{.screen = **screen,
                                   .ui = **ui,
                                   .options = options,
                                   .paths = paths,
                                   .catalog = catalog,
                                   .engines_path = engines_path};
  if (!engines->first_run_done || options.first_run) {
    auto shown = run_engine_screen(engine_screen, *engines, profiles, false, next_key, snapshot);
    if (!shown) return std::unexpected(std::move(shown.error()));
    if (!*shown) return {};
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
  auto library = open_backend(catalog, paths, *engines, probe);
  if (!library) return std::unexpected(std::move(library.error()));

  Player player{.options = options,
                .paths = paths,
                .catalog = catalog,
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
    if (const std::optional<EngineChange>& request = (*outcome)->engines) {
      auto handled = handle_engine_change(player, *request);
      if (!handled) {
        result = std::unexpected(std::move(handled.error()));
        break;
      }
      if (!*handled) break;
    }
    if (const std::optional<ExportJob>& job = (*outcome)->export_job) {
      const ExportOutcome exported = run_export(player, *job);
      if (exported.end == ExportEnd::Quit) break;
      player.notice = exported.notice;
    }
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
