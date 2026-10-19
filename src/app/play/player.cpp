#include "app/play/player.h"

#include <format>
#include <fstream>
#include <optional>
#include <utility>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_keyboard.h>
#include <SDL3/SDL_timer.h>
#include <SDL3/SDL_video.h>

#include "app/play/backend_library.h"
#include "app/play/engine_build.h"
#include "app/play/input.h"
#include "app/play/pipeline.h"
#include "app/play/placebo_screen.h"
#include "app/play/session.h"
#include "app/play/upscale_stage.h"
#include "core/clock.h"
#include "core/log.h"
#include "media/demuxer.h"
#include "media/selection.h"
#include "prepare/sha256.h"

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

// Touche pressée d'office (--keys) : un événement SDL de plus dans la file, comme le clavier.
void press(const std::string& name, SDL_Window* window) {
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
  const std::string label =
      model ? std::format("SPAN ×{} · {} · TensorRT · {} → {}x{}", factor, model->stem().string(), reduce,
                          work.width * factor, work.height * factor)
            : std::format("sans agrandissement · {}", reduce);
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
        return UpscaleStage::start(std::move(stage), UpscaleSetup{.backend = backend->get(),
                                                                  .engine = engine.get(),
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
      const auto input = input_of(event, keys);
      if (!input) continue;
      if (std::holds_alternative<ToggleFullscreen>(*input)) {
        session.toggle_fullscreen();
      } else {
        session.command(std::get<Command>(*input));
      }
    }
    (*pipeline)->pump();
    while (auto message = try_take((*pipeline)->inbox())) session.handle(std::move(*message));
    auto record = session.tick(host_now_us());
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
  out.flush();
  pipeline->reset();
  engine.reset();
  backend->reset();
  screen->reset();
  if (failure) return std::unexpected(std::move(*failure));
  return {};
}

}  // namespace ajp::app::play
