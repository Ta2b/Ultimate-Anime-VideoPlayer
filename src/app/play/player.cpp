#include "app/play/player.h"

#include <algorithm>
#include <array>
#include <chrono>
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

#include "app/play/auto_bench.h"
#include "app/play/auto_config.h"
#include "app/play/backend_library.h"
#include "app/play/build_progress.h"
#include "app/play/chain.h"
#include "app/play/chain_setup.h"
#include "app/play/chain_stats.h"
#include "app/play/display_edid.h"
#include "app/play/engine_build.h"
#include "app/play/engine_downloads.h"
#include "app/play/engine_install.h"
#include "app/play/engine_screen.h"
#include "app/play/engine_switch.h"
#include "app/play/export_run.h"
#include "app/play/first_run.h"
#include "app/play/gmfss_model.h"
#include "app/play/home_screen.h"
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

// Un fichier de configuration que le lecteur n'a pas compris sera réécrit au prochain changement
// de réglage : il est mis de côté d'abord (« nom.invalide-AAAAMMJJ-HHMMSS »), pour qu'aucun profil
// ni réglage valide qu'il contenait ne se perde sans copie. Rien si le fichier est illisible
// (droits) : il n'est pas écrasé non plus. Rend le message à montrer à l'accueil.
std::string set_aside(const std::filesystem::path& path, const Refusal& why) {
  std::error_code error;
  if (why.code == RefusalCode::FileUnreadable || !std::filesystem::exists(path, error)) return {};
  const auto now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
  std::filesystem::path aside = path;
  aside += std::format(".invalide-{:%Y%m%d-%H%M%S}", now);
  std::filesystem::rename(path, aside, error);
  if (error) {
    log_channel("app")->warn("{} : mise de côté impossible ({})", path.string(), error.message());
    return {};
  }
  log_channel("app")->warn("{} mis de côté dans {}", path.string(), aside.string());
  return std::format("{} était invalide : copie gardée sous {}", path.filename().string(),
                     aside.filename().string());
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
  FontSearch fonts = FontSearch::start();  // pendant la lecture des réglages et l'ouverture de l'écran
  // Le fichier donné est lu avant d'ouvrir la fenêtre : un fichier illisible n'en ouvre aucune.
  std::optional<media::TrackInfo> given;
  if (!options.file.empty()) {
    auto video = probe_video(options.file);
    if (!video) return std::unexpected(std::move(video.error()));
    given = std::move(*video);
  }
  const std::filesystem::path settings_path = paths.engines.parent_path() / "play.toml";
  std::vector<std::string> startup_notes;  // fichiers mis de côté, dits à l'accueil
  auto saved = load_settings(settings_path);
  if (!saved) {
    log->warn("{} ; réglages par défaut", saved.error().detail);
    startup_notes.push_back(set_aside(settings_path, saved.error()));
    saved = PlaySettings{};
  }
  // Le profil de démarrage fixe la chaîne ; les options données la remplacent pour cette lecture.
  ProfileState profiles{.user_path = settings_path.parent_path() / "user.toml",
                        .source = given ? core::SourceInfo{.width = given->width,
                                                           .height = given->height,
                                                           .fps = fps_of(given->frame_rate)}
                                        : kHomeSource};
  if (auto loaded = load_profiles(profiles.user_path)) {
    apply_user_keys(loaded->user_text);
    profiles.data = std::move(*loaded);
  } else {
    log->warn("{} ; profils de base", loaded.error().detail);
    startup_notes.push_back(set_aside(profiles.user_path, loaded.error()));
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
  const std::string title = given ? std::format("ajp — {}", options.file.filename().string()) : "ajp";
  auto screen =
      PlaceboScreen::open(ScreenOptions{.title = title,
                                        .width = options.window_width,
                                        .height = options.window_height,
                                        .fullscreen = options.fullscreen,
                                        .hdr = !options.sdr,
                                        .smooth_motion = saved->smooth_motion && saved->rife_factor >= 2,
                                        .allow_software = options.software_gpu});
  if (!screen) return std::unexpected(std::move(screen.error()));
  auto ui = PlayerUi::create((*screen)->window(), given ? options.file.filename().string() : "", fonts);
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
    startup_notes.push_back(set_aside(engines_path, engines.error()));
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
  // Un moteur d'une version précédente du lecteur ne s'ouvrirait pas : il se met à jour d'abord.
  auto updated = update_stale_engines(**screen, **ui, catalog, *engines);
  if (!updated) return std::unexpected(std::move(updated.error()));
  if (!*updated) return {};

  // Lancé sans fichier, l'accueil choisit la vidéo, et il revient à la fin de chacune. Une vidéo
  // glissée sur la fenêtre pendant la lecture prend la place de celle en cours.
  const bool from_home = !given;
  std::filesystem::path file = options.file;
  std::string problem;  // la vidéo choisie ne se lit pas : dit à l'accueil
  for (const std::string& note : startup_notes) {
    if (note.empty()) continue;
    problem += problem.empty() ? note : " · " + note;
  }
  std::shared_ptr<BackendLibrary> library;  // ouverte à la première vidéo, gardée pour les suivantes
  const HomeScreen home{.screen = **screen,
                        .ui = **ui,
                        .options = options,
                        .paths = paths,
                        .catalog = catalog,
                        .settings_path = settings_path};
  while (true) {
    if (file.empty()) {
      HomeState state{.saved = *saved,
                      .profiles = profiles,
                      .engines = *engines,
                      .next_key = next_key,
                      .snapshot = snapshot};
      auto picked = run_home(home, state, problem);
      if (!picked) return std::unexpected(std::move(picked.error()));
      if (!*picked) return {};
      file = std::move(**picked);
    }
    media::TrackInfo video;
    if (given) {
      video = std::move(*given);
      given.reset();
    } else {
      auto probed = probe_video(file);
      if (!probed) {
        log->warn("{}", probed.error().detail);
        problem = std::format("« {} » ne se lit pas : {}", file.filename().string(), probed.error().detail);
        file.clear();
        continue;
      }
      video = std::move(*probed);
      problem.clear();
      // Réglages relus : l'accueil ou la vidéo précédente les ont peut-être changés.
      if (auto reread = load_settings(settings_path)) saved = std::move(*reread);
      SDL_SetWindowTitle((*screen)->window(), std::format("ajp — {}", file.filename().string()).c_str());
      (*ui)->set_title(file.filename().string());
    }
    PlayOptions file_options = options;
    file_options.file = file;
    profiles.source =
        core::SourceInfo{.width = video.width, .height = video.height, .fps = fps_of(video.frame_rate)};

    PlaySettings from_profile = *saved;
    if (const auto& start = profiles.data.file.profiles[static_cast<std::size_t>(profiles.active - 1)]) {
      std::vector<std::string> notes;
      from_profile = apply_profile(*start, profiles.source, *saved, notes);
      for (const std::string& line : notes) log->warn("profil {} : {}", profiles.active, line);
      log->info("profil {} « {} »", profiles.active, start->name);
    }
    // La tranche mesurée par « Évaluer ma machine » (bench.toml), si elle est de cette carte.
    std::optional<BenchResult> measured_tier;
    if (auto bench = load_bench(settings_path.parent_path() / "bench.toml")) measured_tier = std::move(*bench);
    // Le préréglage (Auto, Rapide…) refait la chaîne pour cette carte, cet écran et cette vidéo ;
    // Personnalisé laisse les réglages tels quels.
    from_profile = apply_preset(from_profile,
                                Machine{.tier = effective_tier(gpu_facts(**screen), measured_tier),
                                        .display_height = display_height((*screen)->window()),
                                        .display_hz = display_refresh_hz((*screen)->window())},
                                profiles.source);
    PlaySettings settings = settings_for(from_profile, file_options);
    if (auto valid = validate(settings); !valid) return std::unexpected(std::move(valid.error()));
    static_cast<void>((*ui)->controls().set_volume(settings.volume));
    if (!library) {
      show_loading(**screen, **ui, "Chargement");
      const gpu::BackendInit probe{
          .render = &(*screen)->device(), .cache_dir = paths.engines, .log = backend_log()};
      auto opened = open_backend(catalog, paths, *engines, probe);
      if (!opened) return std::unexpected(std::move(opened.error()));
      library = std::move(*opened);
    }

    Player player{.options = file_options,
                  .paths = paths,
                  .catalog = catalog,
                  .screen = **screen,
                  .ui = **ui,
                  .library = library,
                  .video = video,
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
    std::int64_t start_us = file_options.start_us;
    // settings.toml ([playback] resume_position) l'autorise, et --no-resume l'écarte pour une lecture.
    if (file_options.resume && player.engines.resume_position) {
      player.resume_path = settings_path.parent_path() / "resume.tsv";
      if (start_us == 0) {
        if (const auto at = resume_position(player.resume_path, file)) {
          start_us = *at;
          player.resumed = true;
          log->info("reprise à {}", format_position(start_us));
        }
      }
    }
    std::expected<void, Refusal> result;
    bool start_paused = false;
    std::optional<std::filesystem::path> next_file;
    while (true) {
      auto outcome = run_chain(player, settings, start_us, start_paused);
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
        if (!*handled) {
          player.quit = true;
          break;
        }
      }
      if (const std::optional<ExportJob>& job = (*outcome)->export_job) {
        const ExportOutcome exported = run_export(player, *job);
        if (exported.end == ExportEnd::Quit) {
          player.quit = true;
          break;
        }
        player.notice = exported.notice;
      }
      settings = (*outcome)->settings;
      start_us = (*outcome)->position_us;
      start_paused = (*outcome)->paused;
      player.details = settings.stats.all;
      if ((*outcome)->next_file) {
        if (auto written = save_settings(
                settings_path, settings_to_save(settings, player.from_file, player.started, file_options));
            !written)
          log->warn("{}", written.error().detail);
        player.unsaved.reset();
        next_file = (*outcome)->next_file;
        break;
      }
      player.unsaved = settings;  // écrits quand la chaîne rouverte est ouverte (run_chain)
      log->info("réouverture à {}", format_position(start_us));
    }
    // Une vidéo choisie à l'accueil qui échoue y ramène avec la raison, plutôt que de fermer le
    // lecteur ; lancé sur un fichier, l'échec reste celui du programme (code de sortie, message).
    std::string failed;
    if (!result) {
      if (!from_home || player.quit) return result;
      failed = std::format("« {} » s'est arrêtée : {}", file.filename().string(), result.error().detail);
      log->error("{}", failed);
    }
    // Ce que la lecture a changé suit dans la vidéo suivante ou l'accueil.
    library = std::move(player.library);
    snapshot = std::move(player.snapshot);
    next_key = player.next_key;
    profiles = std::move(player.profiles);
    *engines = std::move(player.engines);
    if (!failed.empty()) {
      problem = std::move(failed);
      file.clear();
    } else if (next_file) {
      file = std::move(*next_file);
    } else if (from_home && !player.quit) {
      file.clear();  // fin de la vidéo : retour à l'accueil
    } else {
      return {};
    }
    if (auto reread = load_settings(settings_path)) saved = std::move(*reread);
  }
}

}  // namespace ajp::app::play
