#include "app/play/player.h"

#include <algorithm>
#include <charconv>
#include <format>
#include <fstream>
#include <optional>
#include <utility>
#include <variant>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_keyboard.h>
#include <SDL3/SDL_properties.h>
#include <SDL3/SDL_timer.h>
#include <SDL3/SDL_video.h>

#include "app/play/backend_library.h"
#include "app/play/engine_build.h"
#include "app/play/input.h"
#include "app/play/pipeline.h"
#include "app/play/placebo_screen.h"
#include "app/play/session.h"
#include "app/play/ui.h"
#include "app/play/upscale_stage.h"
#include "core/clock.h"
#include "core/log.h"
#include "kernels/spec/rife.h"
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
std::expected<media::TrackInfo, Refusal> probe_video(const std::filesystem::path& file) {
  auto demuxer = media::Demuxer::open(file);
  if (!demuxer) return std::unexpected(std::move(demuxer.error()));
  const media::TrackChoice choice = media::choose_tracks(demuxer->tracks(), {});
  if (!choice.video)
    return refuse(RefusalCode::TrackUnavailable, std::format("{} : aucune piste vidéo", file.string()));
  return *demuxer->track(*choice.video);
}

// Tours d'attente de main pendant la construction du moteur : la fenêtre reste vivante et
// dit où en est trtexec. Rien si l'utilisateur ferme la fenêtre.
std::expected<std::optional<gpu::Artifact>, Refusal> wait_engine(EngineBuild& build, PlaceboScreen& screen,
                                                                 const std::string& label) {
  const core::KeyMap keys = core::KeyMap::defaults();
  while (true) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      const auto input = input_of(event, keys);
      if (input && std::holds_alternative<Command>(*input) &&
          std::holds_alternative<cmd::Quit>(std::get<Command>(*input))) {
        return std::nullopt;
      }
    }
    if (auto result = build.result()) {
      if (!*result) return std::unexpected(std::move(result->error()));
      return std::optional(std::move(**result));
    }
    const std::vector<std::string> lines{
        label, "Construction du moteur TensorRT (une fois par taille d'image)", build.progress()};
    auto shown = screen.show(ShowRequest{.osd = lines});
    if (!shown) return std::unexpected(std::move(shown.error()));
    if (!shown->presented) SDL_Delay(10);
  }
}

template <class... F>
struct Overloaded : F... {
  using F::operator()...;
};

// « mouse=XxY » ou « click=XxY » de --keys : la souris passe (et clique) à ce point de la
// fenêtre (fractions de sa taille).
std::optional<std::pair<float, float>> mouse_point(std::string_view text) {
  if (!text.starts_with("mouse=") && !text.starts_with("click=")) return std::nullopt;
  text.remove_prefix(6);
  const std::size_t cross = text.find('x');
  if (cross == std::string_view::npos) return std::nullopt;
  float x = 0.0F;
  float y = 0.0F;
  const std::string_view xs = text.substr(0, cross);
  const std::string_view ys = text.substr(cross + 1);
  if (std::from_chars(xs.data(), xs.data() + xs.size(), x).ec != std::errc{} ||
      std::from_chars(ys.data(), ys.data() + ys.size(), y).ec != std::errc{})
    return std::nullopt;
  return std::pair{x, y};
}

// Touche pressée d'office (--keys) : un événement SDL de plus dans la file, comme le clavier.
// « mouse=XxY » déplace la souris (la barre de lecture paraît), « click=XxY » y clique.
void press(const std::string& name, SDL_Window* window) {
  if (const auto point = mouse_point(name)) {
    int w = 0;
    int h = 0;
    SDL_GetWindowSize(window, &w, &h);
    SDL_Event motion{};
    motion.type = SDL_EVENT_MOUSE_MOTION;
    motion.motion.windowID = SDL_GetWindowID(window);
    motion.motion.x = point->first * static_cast<float>(w);
    motion.motion.y = point->second * static_cast<float>(h);
    log_channel("app")->info("--keys : souris en {:.0f}, {:.0f}", motion.motion.x, motion.motion.y);
    SDL_PushEvent(&motion);
    if (name.starts_with("click=")) {
      for (const bool down : {true, false}) {
        SDL_Event button{};
        button.type = down ? SDL_EVENT_MOUSE_BUTTON_DOWN : SDL_EVENT_MOUSE_BUTTON_UP;
        button.button.windowID = motion.motion.windowID;
        button.button.button = SDL_BUTTON_LEFT;
        button.button.down = down;
        button.button.clicks = 1;
        button.button.x = motion.motion.x;
        button.button.y = motion.motion.y;
        SDL_PushEvent(&button);
      }
    }
    return;
  }
  SDL_Event event{};
  event.type = SDL_EVENT_KEY_DOWN;
  event.key.windowID = SDL_GetWindowID(window);
  event.key.key = SDL_GetKeyFromName(name.c_str());
  event.key.down = true;
  if (event.key.key == SDLK_UNKNOWN) {
    log_channel("app")->warn("--keys : touche inconnue « {} »", name);
    return;
  }
  log_channel("app")->info("--keys : {}", name);
  SDL_PushEvent(&event);
}

// Période de rafraîchissement de l'écran de la fenêtre, d'après SDL ; 0 si inconnue (la session
// la mesure alors sur les présentations).
std::int64_t display_period_us(SDL_Window* window) {
  const SDL_DisplayMode* mode = SDL_GetCurrentDisplayMode(SDL_GetDisplayForWindow(window));
  if (mode == nullptr || mode->refresh_rate <= 0.0F) return 0;
  return static_cast<std::int64_t>(1e6 / static_cast<double>(mode->refresh_rate));
}

// Crête vue par le contenu sur l'écran de la fenêtre : la marge HDR de SDL3 (relative au blanc de
// référence du compositeur) × 203 nits ; rien si l'écran n'est pas en HDR (display.md).
std::optional<double> content_peak_nits(SDL_Window* window) {
  const SDL_PropertiesID props = SDL_GetWindowProperties(window);
  if (!SDL_GetBooleanProperty(props, SDL_PROP_WINDOW_HDR_ENABLED_BOOLEAN, false)) return std::nullopt;
  const float headroom = SDL_GetFloatProperty(props, SDL_PROP_WINDOW_HDR_HEADROOM_FLOAT, 0.0F);
  if (!(headroom > 1.0F)) return std::nullopt;
  return static_cast<double>(headroom) * 203.0;
}

// RTX HDR : le SDK Maxine importé une fois dans le magasin du lecteur (environ 135 Mo), revérifié
// à chaque lancement, puis l'effet ouvert pour la sortie de SPAN ; l'écran reçoit alors du PQ
// BT.2020 au pic donné à TrueHDR.
std::expected<void, Refusal> open_rtx_hdr(const PlayOptions& options, const PlayerPaths& paths,
                                          PlaceboScreen& screen, gpu::KernelOps& kernels, int width,
                                          int height) {
  const auto log = log_channel("app");
  vendor_hdr::SdkStore store(paths.engines.parent_path() / "sdk", vendor_hdr::SdkKind::MaxineVfx);
  vendor_hdr::SdkStatus sdk = store.status();
  if (!sdk.installed) {
    log->info("RTX HDR : import du SDK Maxine depuis {} (environ 135 Mo, une seule fois)",
              options.true_hdr_sdk.string());
    auto imported = store.import(options.true_hdr_sdk);
    if (!imported) return std::unexpected(std::move(imported.error()));
    sdk = *imported;
  }
  if (!sdk.installed) return refuse(RefusalCode::HdrUnavailable, std::format("RTX HDR : {}", sdk.reason));
  const int peak = options.true_hdr_peak > 0
                       ? options.true_hdr_peak
                       : vendor_hdr::peak_from_display(content_peak_nits(screen.window()));
  auto params = vendor_hdr::validate(vendor_hdr::TrueHdrParams{.contrast = options.true_hdr_contrast,
                                                               .saturation = options.true_hdr_saturation,
                                                               .middle_gray = options.true_hdr_gray,
                                                               .peak_nits = peak});
  if (!params) return std::unexpected(std::move(params.error()));
  if (auto ok = kernels.open_true_hdr(
          *sdk.installed,
          vendor_hdr::TrueHdrConfig{
              .width = width, .height = height, .source = ColorInfo{}, .params = *params});
      !ok) {
    return ok;
  }
  screen.set_source_color(ColorInfo{.matrix = ColorMatrix::Rgb,
                                    .range = ColorRange::Full,
                                    .transfer = Transfer::Pq,
                                    .primaries = Primaries::Bt2020,
                                    .hdr = HdrMetadata{.max_luminance = static_cast<float>(peak)}});
  log->info("RTX HDR : Maxine {} · pic {} nits · contraste {} · saturation {} · gris moyen {}",
            sdk.installed->version().text(), peak, params->contrast, params->saturation, params->middle_gray);
  return {};
}

std::string volume_text(const Controls& controls) {
  if (controls.muted()) return "Son coupé";
  return std::format("Volume {} %", std::lround(controls.volume() * 100.0F));
}

std::string label_of(const std::vector<TrackEntry>& entries, int track) {
  const auto found = std::ranges::find(entries, track, &TrackEntry::index);
  return found != entries.end() ? found->label : std::format("piste {}", track);
}

// Ce que les actions touchent : la session, l'écran, l'interface et les pistes du pipeline.
struct ActionTargets {
  Session& session;
  PlaceboScreen& screen;
  PlayerUi& ui;
  Pipeline& pipeline;
};

// Piste audio : le pont change de piste à l'époque du seek qui suit, à la position courante.
void switch_audio(const ActionTargets& to, int track, std::int64_t now_us) {
  if (to.pipeline.tracks().audio_selected == track) return;
  const std::int64_t position = std::max<std::int64_t>(to.session.view().position_us, 0);
  auto switched = to.pipeline.select_audio(track, to.session.epoch() + 1, [&] {
    to.session.command(cmd::Seek{.target_us = position, .exact = true});
  });
  if (!switched) {
    log_channel("app")->warn("piste audio {} : {}", track, switched.error().detail);
    to.ui.controls().notify("Piste audio refusée", now_us);
    return;
  }
  to.ui.set_tracks(to.pipeline.tracks());
  to.ui.controls().notify(std::format("Audio : {}", label_of(to.pipeline.tracks().audio, track)), now_us);
}

// Sous-titres : subs prend la nouvelle piste, un seek à la position courante relit ses paquets.
void switch_subtitles(const ActionTargets& to, std::optional<int> track, std::int64_t now_us) {
  if (to.pipeline.tracks().subtitle_selected == track) return;
  auto switched = to.pipeline.select_subtitles(track);
  if (!switched) {
    log_channel("app")->warn("sous-titres : {}", switched.error().detail);
    to.ui.controls().notify("Piste de sous-titres refusée", now_us);
    return;
  }
  to.session.set_subtitles(track.has_value());
  if (track) {
    const std::int64_t position = std::max<std::int64_t>(to.session.view().position_us, 0);
    to.session.command(cmd::Seek{.target_us = position, .exact = true});
  }
  to.ui.set_tracks(to.pipeline.tracks());
  to.ui.controls().notify(
      track ? std::format("Sous-titres : {}", label_of(to.pipeline.tracks().subtitles, *track))
            : std::string("Sous-titres désactivés"),
      now_us);
}

// Une action du clavier ou de l'interface. La barre paraît aussi après un seek au clavier.
void apply(const Input& input, const ActionTargets& to, std::int64_t now_us) {
  Session& session = to.session;
  PlaceboScreen& screen = to.screen;
  Controls& controls = to.ui.controls();
  std::visit(
      Overloaded{
          [&](const Command& command) {
            if (std::holds_alternative<cmd::TogglePause>(command)) {
              log_channel("app")->info("action : {}", session.paused() ? "lecture" : "pause");
            } else if (const auto* seek = std::get_if<cmd::Seek>(&command)) {
              log_channel("app")->info("action : seek {}{}", format_position(seek->target_us),
                                       seek->exact ? "" : " (point clé)");
            }
            session.command(command);
            if (std::holds_alternative<cmd::SeekRelative>(command)) controls.activity(now_us);
            if (std::holds_alternative<cmd::ToggleOsd>(command)) {
              controls.notify(session.view().stats ? "Statistiques affichées" : "Statistiques masquées",
                              now_us);
            }
          },
          [&](const ToggleFullscreen&) { session.toggle_fullscreen(); },
          [&](const LeaveFullscreen&) {
            if (screen.fullscreen()) session.toggle_fullscreen();
          },
          [&](const ToggleMute&) {
            session.command(Command{controls.toggle_mute()});
            controls.notify(volume_text(controls), now_us);
          },
          [&](const VolumeStep& step) {
            session.command(Command{controls.step_volume(step.delta)});
            controls.notify(volume_text(controls), now_us);
          },
          [&](const SelectAudio& select) { switch_audio(to, select.track, now_us); },
          [&](const SelectSubtitles& select) { switch_subtitles(to, select.track, now_us); },
          [&](const CycleAudio&) {
            if (auto next = next_audio(to.pipeline.tracks())) switch_audio(to, *next, now_us);
          },
          [&](const CycleSubtitles&) { switch_subtitles(to, next_subtitle(to.pipeline.tracks()), now_us); }},
      input);
}

void write_record(std::ofstream& stats, const PresentRecord& record) {
  stats << record.host_us << '\t' << record.pts_us << '\t' << (record.fresh ? 1 : 0) << '\t'
        << record.render_us << '\t' << record.interval_us << '\t' << record.dropped << '\n';
}

void log_summary(const Session& session) {
  const FrameStats& stats = session.stats();
  log_channel("app")->info(
      "fin : {} images neuves, {} présentations, {} sautées, {} jetées au seek, {} sans sous-titres en "
      "attente ; "
      "temps par image médian {} (σ {}), p95 {} ; composition médiane {}",
      stats.new_frames, stats.presents, stats.dropped, stats.stale, stats.subtitle_gaps,
      format_ms(stats.frame_interval.median()),
      format_ms(static_cast<std::int64_t>(stats.frame_interval.stddev())),
      format_ms(stats.frame_interval.percentile(95)), format_ms(stats.render.median()));
}

}  // namespace

std::expected<void, Refusal> play(const PlayOptions& options, const PlayerPaths& paths) {
  const auto log = log_channel("app");
  auto video = probe_video(options.file);
  if (!video) return std::unexpected(std::move(video.error()));
  std::optional<std::filesystem::path> model;
  if (options.upscale) {
    auto found = find_model(options, default_model_candidates(paths.home));
    if (!found) return std::unexpected(std::move(found.error()));
    model = *found;
  }
  const int factor = model ? kSpanFactor : 1;
  const WorkSize work = work_size(video->width, video->height, options.work_height);
  const std::string reduce =
      work == WorkSize{video->width, video->height}
          ? std::format("{}x{}", video->width, video->height)
          : std::format("{}x{} → Area {}x{}", video->width, video->height, work.width, work.height);
  std::optional<std::filesystem::path> rife_model;
  if (options.rife_factor >= 2) {
    auto found = find_rife_model(options, default_rife_candidates(paths.home));
    if (!found) return std::unexpected(std::move(found.error()));
    rife_model = *found;
  }
  const std::string rife_label =
      (rife_model ? std::format("RIFE ×{} · {} · ", options.rife_factor, rife_model->stem().string()) : "") +
      (options.true_hdr ? std::string("RTX HDR · ") : "");
  const std::string label =
      model ? std::format("{}SPAN ×{} · {} · TensorRT · {} → {}x{}", rife_label, factor,
                          model->stem().string(), reduce, work.width * factor, work.height * factor)
            : std::format("{}sans agrandissement · {}", rife_label, reduce);
  log->info("{} : {}", options.file.string(), label);

  auto screen =
      PlaceboScreen::open(ScreenOptions{.title = std::format("ajp — {}", options.file.filename().string()),
                                        .width = options.window_width,
                                        .height = options.window_height,
                                        .fullscreen = options.fullscreen});
  if (!screen) return std::unexpected(std::move(screen.error()));
  auto library = BackendLibrary::open(paths.backends);
  if (!library) return std::unexpected(std::move(library.error()));
  const gpu::BackendInit init{
      .render = &(*screen)->device(), .cache_dir = paths.engines, .log = backend_log()};
  auto backend = BackendLibrary::create(*library, init);
  if (!backend) return std::unexpected(std::move(backend.error()));
  log->info("backend {} ({}) depuis {}", (*backend)->capabilities().name,
            (*backend)->capabilities().device_name, (*library)->path().string());

  std::unique_ptr<gpu::Engine> engine;
  if (model) {
    auto sha256 = prepare::sha256_file(*model);
    if (!sha256) return std::unexpected(std::move(sha256.error()));
    const gpu::Shape shape{.dims = {1, 3, work.height, work.width}};
    auto build = EngineBuild::start(
        EngineRequest{.library = *library,
                      .init = init,
                      .model = {.id = model->stem().string(), .onnx_path = *model, .sha256 = *sha256},
                      .shapes = {.min = shape, .opt = shape, .max = shape}});
    if (!build) return std::unexpected(std::move(build.error()));
    auto artifact = wait_engine(**build, **screen, label);
    if (!artifact) return std::unexpected(std::move(artifact.error()));
    if (!*artifact) return {};  // fenêtre fermée pendant la construction
    log->info("moteur {}", (*artifact)->path.string());
    auto loaded = (*backend)->load(**artifact);
    if (!loaded) return std::unexpected(std::move(loaded.error()));
    engine = std::move(*loaded);
  }
  std::unique_ptr<gpu::Engine> rife;
  if (rife_model) {
    auto sha256 = prepare::sha256_file(*rife_model);
    if (!sha256) return std::unexpected(std::move(sha256.error()));
    const kernels::RifeGeometry g = kernels::rife_geometry(work.width, work.height);
    const gpu::Shape shape{.dims = {1, kernels::kRifeInputChannels, g.padded_height, g.padded_width}};
    auto build = EngineBuild::start(EngineRequest{
        .library = *library,
        .init = init,
        .model = {.id = rife_model->stem().string(), .onnx_path = *rife_model, .sha256 = *sha256},
        .shapes = {.min = shape, .opt = shape, .max = shape}});
    if (!build) return std::unexpected(std::move(build.error()));
    auto artifact = wait_engine(**build, **screen, label);
    if (!artifact) return std::unexpected(std::move(artifact.error()));
    if (!*artifact) return {};  // fenêtre fermée pendant la construction
    log->info("moteur RIFE {}", (*artifact)->path.string());
    auto loaded = (*backend)->load(**artifact);
    if (!loaded) return std::unexpected(std::move(loaded.error()));
    rife = std::move(*loaded);
  }

  if (options.true_hdr) {
    auto opened = open_rtx_hdr(options, paths, **screen, *(*backend)->kernels(), work.width * factor,
                               work.height * factor);
    if (!opened) return std::unexpected(std::move(opened.error()));
  }

  PipelineConfig config{
      .file = options.file,
      .decoder = {.attempts = {media::HwApi::Cuda}, .render_device = (*screen)->device().uuid}};
  PlaceboScreen& out = **screen;
  auto pipeline = Pipeline::open(
      config, [&](VideoStageInit stage) -> std::expected<std::unique_ptr<VideoStage>, Refusal> {
        const int width = stage.track->width;
        const int height = stage.track->height;
        const WorkSize sized = work_size(width, height, options.work_height);
        auto slots = out.create_ring(sized.width * factor, sized.height * factor, kRingSlots);
        if (!slots) return std::unexpected(std::move(slots.error()));
        return UpscaleStage::start(
            std::move(stage),
            UpscaleSetup{.backend = backend->get(),
                         .engine = engine.get(),
                         .rife = rife.get(),
                         .rife_factor = static_cast<std::uint32_t>(std::max(options.rife_factor, 1)),
                         .true_hdr = options.true_hdr,
                         .source_width = width,
                         .source_height = height,
                         .work_width = sized.width,
                         .work_height = sized.height,
                         .slots = std::move(*slots)});
      });
  if (!pipeline) return std::unexpected(std::move(pipeline.error()));

  SessionConfig session_config = (*pipeline)->session_config();
  session_config.start_us = options.start_us;
  session_config.stop_after_us = options.stop_after_us;
  session_config.fullscreen_after_us = options.fullscreen_after_us;
  session_config.header = {label};
  session_config.display_period_us = display_period_us((*screen)->window());
  for (std::string& line : (*pipeline)->describe()) session_config.header.push_back(std::move(line));
  session_config.header.push_back(out.output_description());
  for (const std::string& line : session_config.header) log->info("{}", line);
  Session session(std::move(session_config), (*pipeline)->stages(), out, (*pipeline)->clock());
  auto ui = PlayerUi::create(out.window(), options.file.filename().string());
  if (!ui) return std::unexpected(std::move(ui.error()));
  (*ui)->set_tracks((*pipeline)->tracks());
  const ActionTargets targets{.session = session, .screen = out, .ui = **ui, .pipeline = **pipeline};

  std::ofstream stats;
  if (options.stats) {
    stats.open(*options.stats);
    stats << "host_us\tpts_us\tfresh\trender_us\tinterval_us\tdropped\n";
  }
  const core::KeyMap keys = core::KeyMap::defaults();
  std::optional<Refusal> failure;
  const std::int64_t began_us = host_now_us();
  std::optional<std::filesystem::path> snapshot = options.snapshot;
  std::size_t next_key = 0;
  std::vector<Input> actions;
  session.begin(began_us);
  while (!session.finished()) {
    for (; next_key < options.keys.size() && host_now_us() - began_us >= options.keys[next_key].first;
         ++next_key) {
      press(options.keys[next_key].second, (*screen)->window());
    }
    if (snapshot && host_now_us() - began_us >= options.snapshot_at_us) {
      out.snapshot_next(*snapshot);
      snapshot.reset();
    }
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if ((*ui)->handle(event, host_now_us(), actions)) continue;  // saisie dans l'interface
      if (auto input = input_of(event, keys)) actions.push_back(*input);
    }
    (*pipeline)->pump();
    while (auto message = try_take((*pipeline)->inbox())) session.handle(std::move(*message));
    const UiFrame ui_frame = (*ui)->frame(host_now_us(), session.view(), actions);
    for (const Input& input : actions) apply(input, targets, host_now_us());
    actions.clear();
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
  log_summary(session);
  for (const std::string& refusal : session.refusals()) log->warn("{}", refusal);
  // Ordre d'arrêt : les slots rendus, les threads arrêtés, le moteur avant son backend, le
  // backend (et ses imports du ring) avant l'écran qui possède le ring.
  session.end();
  (*ui)->release(out);
  out.flush();
  pipeline->reset();
  rife.reset();
  engine.reset();
  backend->reset();
  screen->reset();
  if (failure) return std::unexpected(std::move(*failure));
  return {};
}

}  // namespace ajp::app::play
