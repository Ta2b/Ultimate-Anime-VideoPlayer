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
#include "app/play/player_run.h"
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
