#include "app/play/player.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <format>
#include <fstream>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <variant>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_timer.h>
#include <SDL3/SDL_video.h>

#include "app/play/auto_bench.h"
#include "app/play/auto_config.h"
#include "app/play/backend_library.h"
#include "app/play/bench_events.h"
#include "app/play/build_progress.h"
#include "app/play/chain.h"
#include "app/play/chain_setup.h"
#include "app/play/chain_stats.h"
#include "app/play/display_edid.h"
#include "app/play/engine_build.h"
#include "app/play/engine_downloads.h"
#include "app/play/engine_install.h"
#include "app/play/engine_online.h"
#include "app/play/engine_screen.h"
#include "app/play/engine_switch.h"
#include "app/play/export_run.h"
#include "app/play/first_run.h"
#include "app/play/home_screen.h"
#include "app/play/host_facts.h"
#include "app/play/i18n.h"
#include "app/play/input.h"
#include "app/play/look_catalog.h"
#include "app/play/machine_calibration.h"
#include "app/play/model_choice.h"
#include "app/play/paths.h"
#include "app/play/pipeline.h"
#include "app/play/placebo_screen.h"
#include "app/play/player_engines.h"
#include "app/play/player_run.h"
#include "app/play/pumped_task.h"
#include "app/play/resume.h"
#include "app/play/scripted_input.h"
#include "app/play/session.h"
#include "app/play/ui.h"
#include "app/play/updater.h"
#include "app/play/upscale_stage.h"
#include "app/play/user_keys.h"
#include "app/play/visible_error.h"
#include "core/clock.h"
#include "core/log.h"
#include "core/settings.h"
#include "core/thread.h"
#include "media/demuxer.h"
#include "media/selection.h"
#include "prepare/sha256.h"
#include "prepare/utf8.h"

namespace ajp::app::play {
namespace {

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
  return i18n::tr_format("{} was invalid: copy kept as {}", path.filename().string(),
                         aside.filename().string());
}

// La piste vidéo que la source choisira, lue d'avance : sa taille fixe le moteur et le ring.
std::expected<media::TrackInfo, Refusal> probe_once(const std::filesystem::path& file) {
  auto demuxer = media::Demuxer::open(file);
  if (!demuxer) return std::unexpected(std::move(demuxer.error()));
  const media::TrackChoice choice = media::choose_tracks(demuxer->tracks(), {});
  if (!choice.video) {
    // P3.28 : un fichier audio seul est dit tel quel, pas comme un fichier abîmé.
    Refusal refusal{RefusalCode::TrackUnavailable, std::format("{} : aucune piste vidéo", file.string())};
    if (choice.audio) refusal.key = kAudioOnlyKey;
    return std::unexpected(std::move(refusal));
  }
  const media::TrackInfo& video = *demuxer->track(*choice.video);
  // P3.26 : le profil 5 n'a pas de couche de base HDR10 ou SDR (IPTPQc2) ; les profils 7 et 8 se lisent par
  // leur couche de base, sans les métadonnées dynamiques.
  if (video.dolby_vision_profile == 5) {
    Refusal refusal{RefusalCode::FormatUnsupported, std::format("{} : Dolby Vision profil 5", file.string())};
    refusal.key = kDolbyVision5Key;
    return std::unexpected(std::move(refusal));
  }
  if (video.dolby_vision_profile > 0)
    log_channel("app")->info("Dolby Vision profil {} : lu par sa couche de base, sans RPU",
                             video.dolby_vision_profile);
  return video;
}

// Hors du thread de la fenêtre, borné (P2.15) : une vidéo sur un partage réseau qui ne répond plus laisse la
// fenêtre vivante et rend la main avec une raison.
constexpr auto kProbeLimit = std::chrono::seconds(20);
std::expected<media::TrackInfo, Refusal> probe_video(const std::filesystem::path& file, bool window_open) {
  return run_pumped_for<media::TrackInfo>("probe", kProbeLimit, window_open,
                                          [file] { return probe_once(file); });
}

// Les moteurs téléchargeables : ceux de l'engines.lock livré, et ceux joints à la release après elle
// (TensorRT construit plus tard), relus en ligne sauf si les mises à jour sont refusées (la copie gardée sert
// alors) ou pour un dossier de données à part (essais).
EngineCatalog engine_catalog(const PlayOptions& options, const PlayerPaths& paths,
                             const PlaySettings& saved) {
  EngineCatalog catalog = load_catalog(
      paths.engine_index, paths.engines.parent_path(),
      saved.engines_dir.empty() ? std::filesystem::path() : prepare::path_from_utf8(saved.engines_dir));
  const bool online = saved.update_mode != UpdateMode::Never && !options.data_dir;
  add_release_engines(catalog, paths.engines.parent_path() / "engines-nvidia.lock", release_repo(),
                      installed_version(), online ? UpdateFetch(fetch_https) : UpdateFetch());
  if (!catalog.problem.empty()) log_channel("app")->info("{}", catalog.problem);
  return catalog;
}

// A25-01 : l'assistant relancé en lecture a écrit play.toml (langues, mises à jour, chaîne calibrée). La
// vidéo se rouvre comme un fichier qu'on ouvre, au même endroit : réglages relus, préréglage rejoué, et
// l'ancienne chaîne en mémoire ne réécrit plus ses choix (Reopen::settings_written).
void follow_wizard(const EngineChange& request, Reopen& reopen, const std::filesystem::path& file) {
  const auto* open = std::get_if<OpenEngineScreen>(&request);
  if (open == nullptr || !open->wizard) return;
  reopen.next_file = file;
  reopen.next_start_us = reopen.position_us;
  reopen.next_paused = reopen.paused;
  reopen.settings_written = true;
}

// Avant une autre vidéo : la chaîne en cours dans play.toml, sans les raccourcis de séance (A08-04).
void save_before_next(const Player& player, const Reopen& reopen, const PlaySettings& settings,
                      const PlayOptions& file_options) {
  if (reopen.settings_written) return;
  const PlaySettings kept = with_kept_choices(player.kept, settings);
  if (auto written = save_settings(player.settings_path,
                                   settings_to_save(kept, player.from_file, player.started, file_options));
      !written)
    log_channel("app")->warn("{}", written.error().detail);
}

// La machine à laquelle une calibration se compare (A25-02, A07-05). Le moteur qui jouera SPAN et RIFE : la
// bibliothèque déjà ouverte si le choix n'a pas bougé depuis (opened_for), sinon le préféré s'il est
// installé.
CalibrationHost calibration_host(const GpuFacts& here, int display_height, const BackendLibrary* library,
                                 const std::optional<core::EngineId>& opened_for,
                                 const core::Settings& engines, const EngineCatalog& catalog,
                                 const PlayerPaths& paths) {
  const std::optional<core::EngineId> preferred =
      engines.preferred[static_cast<std::size_t>(core::Usage::Upscale)];
  const bool tensorrt =
      library != nullptr && preferred == opened_for
          ? engine_of_library(library->path()) == core::EngineId::TensorRt
          : preferred == core::EngineId::TensorRt && tensorrt_backend_present(catalog, paths);
  return {.gpu = here.name, .display_height = display_height, .tensorrt = tensorrt, .driver = here.driver};
}

}  // namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity,readability-function-size)
std::expected<void, Refusal> play(const PlayOptions& options, const PlayerPaths& paths) {
  const auto log = log_channel("app");
  set_bench_child(options.bench_out.has_value());  // l'essai d'un banc parle à son parent (bench_events.h)
  FontSearch fonts = FontSearch::start();  // pendant la lecture des réglages et l'ouverture de l'écran
  // Le fichier donné est lu avant d'ouvrir la fenêtre : un fichier illisible n'en ouvre aucune.
  std::optional<media::TrackInfo> given;
  std::string unreadable;  // le fichier donné ne se lit pas : dit à l'accueil, qui s'ouvre à sa place
  if (!options.file.empty()) {
    auto video = probe_video(options.file, false);
    if (video) {
      given = std::move(*video);
    } else if (options.hidden || options.bench_out || options.snapshot || !options.keys.empty()) {
      return std::unexpected(std::move(video.error()));  // un essai scripté : le code de sortie le dit
    } else {
      // Lancé depuis le gestionnaire de fichiers (« Ouvrir avec »), sans terminal : un refus silencieux
      // ne dirait rien. L'accueil s'ouvre avec la raison.
      log_channel("app")->warn("{}", video.error().detail);
      unreadable = i18n::tr_format("\"{}\" cannot be read: {}", options.file.filename().string(),
                                   visible_error(video.error()));
    }
  }
  // Vérification des mises à jour, en arrière-plan et hors lecture : le fil s'arrête avec le lecteur. Au
  // lancement si elle est due, puis relue toutes les 30 minutes tant que le lecteur reste ouvert (la
  // fréquence « au lancement » ne revérifie pas) ; pendant une lecture, elle attend la fin de la vidéo.
  std::atomic<bool> playing{false};
  std::jthread update_check;
  const std::filesystem::path update_file = paths::update_state(paths.data);
  const UpdatePaths update_paths{
      .data = paths.data, .install = paths.executable.parent_path(), .executable = paths.executable};
  const std::filesystem::path settings_path = paths::play_settings(paths::settings_dir(paths));
  std::vector<std::string> startup_notes;  // fichiers mis de côté, dits à l'accueil
  auto saved = load_settings(settings_path);
  if (saved) i18n::set_language(i18n::language_from_code(saved->language));
  if (!saved) {
    log->warn("{} ; réglages par défaut", saved.error().detail);
    startup_notes.push_back(set_aside(settings_path, saved.error()));
    saved = PlaySettings{};
  }
  if (!options.bench_out && !options.data_dir) {
    if (saved->update_mode != UpdateMode::Never) {
      auto started = start_thread(
          "updates", ThreadReport([](const Refusal& refused) {
            log_channel("app")->warn("mise à jour : {}", refused.detail);
          }),
          [=, &playing, mode = saved->update_mode, channel = saved->update_channel,
           frequency = saved->update_frequency](const std::stop_token& stop, const ThreadReport&) {
            std::mutex mutex;
            std::condition_variable_any wake;
            for (bool first = true; !stop.stop_requested(); first = false) {
              const std::int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
                                           std::chrono::system_clock::now().time_since_epoch())
                                           .count();
              UpdateState state = load_update_state(update_file);
              if ((first || !playing.load()) && update_due(state, mode, frequency, now)) {
                const auto message = run_update_check(update_paths, mode, name(channel), state, stop);
                if (const auto kept = save_update_state(update_file, state); !kept)
                  log_channel("app")->warn("update.toml : {}", kept.error().detail);
                log_channel("app")->info("mise à jour : {}", message ? *message : message.error().detail);
              }
              if (frequency == UpdateFrequency::Launch) return;
              std::unique_lock lock(mutex);
              wake.wait_for(lock, stop, std::chrono::minutes(30), [] { return false; });
            }
          });
      if (started)
        update_check = std::move(*started);
      else
        log->warn("mise à jour : {}", started.error().detail);
    }
  }
  load_user_keys(paths::user_keys(settings_path.parent_path()));  // [keys] seul
  const std::string title = given ? std::format("ajp — {}", options.file.filename().string()) : "ajp";
  auto screen = PlaceboScreen::open(ScreenOptions{
      .title = title,
      .width = options.hidden ? 16 : options.window_width,
      .height = options.hidden ? 9 : options.window_height,
      .fullscreen = options.fullscreen && !options.hidden,
      .hidden = options.hidden,
      .offscreen = options.hidden,
      .offscreen_width = options.window_width,
      .offscreen_height = options.window_height,
      .hdr = !options.sdr,
      .kde_sdr_nits = options.kde_sdr_nits,
      .kde_sdr_auto = options.kde_sdr_auto,
      .smooth_motion = saved->smooth_motion && saved->rife_factor >= 2,
      .allow_software = options.software_gpu,
      .device_name = options.software_gpu ? std::string() : saved->gpu,
      .present = options.present,
      .shader_cache = paths::shader_cache(paths),
      .screen_stamps = options.stats ? std::filesystem::path(*options.stats).replace_extension(".screen.tsv")
                                     : std::filesystem::path()});
  if (!screen) return std::unexpected(std::move(screen.error()));
  auto ui = PlayerUi::create((*screen)->window(), given ? options.file.filename().string() : "", fonts);
  if (!ui) return std::unexpected(std::move(ui.error()));
  note_ignored_keys((*ui)->controls());
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
  const std::filesystem::path engines_path = paths::engine_settings(settings_path.parent_path());
  std::vector<std::string> engines_skipped;  // les valeurs écartées une à une (P2.21)
  auto engines = core::load_settings(engines_path, &engines_skipped);
  for (const std::string& skipped : engines_skipped) log->warn("settings.toml : {} écarté", skipped);
  if (!engines) {
    log->warn("{} ; écran de bienvenue", engines.error().detail);
    startup_notes.push_back(set_aside(engines_path, engines.error()));
    engines = core::Settings{};
  }
  // Un moteur débranché depuis (MIGraphX, OpenVINO) sort des choix : Vulkan prend sa place.
  for (const core::EngineId id : {core::EngineId::MiGraphX, core::EngineId::OpenVino}) {
    if (!chosen(*engines, id)) continue;
    log->info("moteurs : {} débranché de ce lecteur, Vulkan à sa place", engine_label(id));
    *engines = without_engine(std::move(*engines), id);
  }
  const EngineCatalog catalog = engine_catalog(options, paths, *saved);
  const EngineScreen engine_screen{.screen = **screen,
                                   .ui = **ui,
                                   .options = options,
                                   .paths = paths,
                                   .catalog = catalog,
                                   .engines_path = engines_path};
  // L'assistant s'ouvre au premier lancement dans tous les cas, double-clic compris (P3.1) : la vidéo
  // demandée attend, puis se lance à la fin. Seuls les lancements pilotés (essais, captures, banc, dossier de
  // réglages à part) le sautent.
  const bool scripted = options.stop_after_us > 0 || options.snapshot || !options.keys.empty() ||
                        options.bench_out || options.data_dir;
  if (options.first_run || (!engines->first_run_done && !scripted)) {
    auto shown = run_engine_screen(engine_screen, *engines, false, next_key, snapshot);
    if (!shown) return std::unexpected(std::move(shown.error()));
    if (!*shown) return {};
    // L'assistant vient d'écrire play.toml (HDR, chaîne calibrée, auto Rapide) : l'accueil part de là, sans
    // quoi il reprendrait les réglages d'avant et les réécrirait par-dessus.
    if (auto reread = load_settings(settings_path)) saved = std::move(*reread);
  }
  // Un moteur d'une version précédente du lecteur ne s'ouvrirait pas : il se met à jour d'abord.
  // Une seule relance par mise à jour : si le module reste périmé après elle, on ne boucle pas.
  const std::filesystem::path relaunch_marker = paths.data / "engines-relaunched";
  std::error_code marker_error;
  const bool relaunched = std::filesystem::remove(relaunch_marker, marker_error);
  bool engines_updated = false;
  auto updated = update_stale_engines(**screen, **ui, catalog, *engines, engines_updated);
  // Le moteur vérifié ici : un autre choisi ensuite à l'accueil se vérifie à son ouverture.
  const std::optional<core::EngineId> checked_engine =
      engines->preferred[static_cast<std::size_t>(core::Usage::Upscale)];
  if (!updated) return std::unexpected(std::move(updated.error()));
  if (!*updated) return {};
  if (engines_updated && !relaunched && paths.relaunch != nullptr) {
    // Le module remplacé ne se charge pas sous le même nom dans ce processus : le lecteur repart, et le
    // moteur choisi s'ouvre au lancement suivant sans autre geste.
    if (std::ofstream(relaunch_marker) << "1\n") log->info("moteurs mis à jour : le lecteur se relance");
    *paths.relaunch = true;
    return {};
  }

  // Lancé sans fichier, l'accueil choisit la vidéo, et il revient à la fin de chacune. Une vidéo
  // glissée sur la fenêtre pendant la lecture prend la place de celle en cours.
  const bool from_home = !given;
  std::filesystem::path file = unreadable.empty() ? options.file : std::filesystem::path();
  std::string problem = unreadable;  // la vidéo choisie ne se lit pas : dit à l'accueil
  std::string set_aside_notes;       // play.toml ou settings.toml mis de côté
  for (const std::string& note : startup_notes) {
    if (note.empty()) continue;
    set_aside_notes += set_aside_notes.empty() ? note : " · " + note;
  }
  // À l'accueil, sous ses boutons ; en lecture directe, sur l'image de la première vidéo (P2.23).
  if (from_home && !set_aside_notes.empty())
    problem += problem.empty() ? set_aside_notes : " · " + set_aside_notes;
  if (from_home) set_aside_notes.clear();
  std::shared_ptr<BackendLibrary> library;       // ouverte à la première vidéo, gardée pour les suivantes
  std::optional<core::EngineId> library_wanted;  // le moteur préféré quand elle a été ouverte
  // La pré-analyse des plans (chain_shots.h) et son cache : seule la vidéo lue s'analyse.
  ShotWork shots{.cache_dir = paths::shot_cache(paths)};
  if (options.probe_nvdec) shots.probe.cuvid_resize = media::kProbeCuvidSize;  // 8.8
  // PlayFile : l'instant et la pause du fichier suivant, à la place de la reprise.
  std::optional<std::int64_t> forced_start;
  bool forced_paused = false;
  // Annoncés une fois par lancement, pas à chaque vidéo : la mise à jour vue, la calibration d'une autre
  // carte.
  bool update_announced = false;
  bool calibration_told = false;
  const HomeScreen home{.screen = **screen,
                        .ui = **ui,
                        .options = options,
                        .paths = paths,
                        .catalog = catalog,
                        .settings_path = settings_path};
  while (true) {
    if (file.empty()) {
      HomeState state{.saved = *saved, .engines = *engines, .next_key = next_key, .snapshot = snapshot};
      auto picked = run_home(home, state, problem);
      if (!picked) return std::unexpected(std::move(picked.error()));
      if (!*picked) return {};
      file = std::move(**picked);
      if (state.start) {
        forced_start = state.start->start_us;
        forced_paused = state.start->paused;
      }
    }
    media::TrackInfo video;
    if (given) {
      video = std::move(*given);
      given.reset();
    } else {
      auto probed = probe_video(file, true);
      if (!probed) {
        log->warn("{}", probed.error().detail);
        problem = i18n::tr_format("\"{}\" cannot be read: {}", file.filename().string(),
                                  visible_error(probed.error()));
        file.clear();
        continue;
      }
      video = std::move(*probed);
      problem.clear();
      // Réglages relus : l'accueil ou la vidéo précédente les ont peut-être changés.
      if (auto reread = load_settings(settings_path)) {
        saved = std::move(*reread);
        i18n::set_language(i18n::language_from_code(saved->language));
      }
      SDL_SetWindowTitle((*screen)->window(), std::format("ajp — {}", file.filename().string()).c_str());
      (*ui)->set_title(file.filename().string());
    }
    PlayOptions file_options = options;
    file_options.file = file;
    const core::SourceInfo source{
        .width = video.width, .height = video.height, .fps = fps_of(video.frame_rate)};

    PlaySettings chosen = *saved;
    // La tranche mesurée par « Évaluer ma machine » (bench.toml), si elle est de cette carte.
    std::optional<BenchResult> measured_tier;
    if (auto bench = load_bench(paths::bench(settings_path.parent_path()))) measured_tier = std::move(*bench);
    // Le préréglage (Auto, Rapide…) refait la chaîne pour cette carte, cet écran et cette vidéo ;
    // Personnalisé laisse les réglages tels quels.
    Tier tier = effective_tier(gpu_facts(**screen), measured_tier);
    if (file_options.bench_level) {
      // Un essai du benchmark : la chaîne du niveau demandé (ou de l'estimé), sans priorité ni profil.
      if (*file_options.bench_level >= 0) tier = static_cast<Tier>(*file_options.bench_level);
      file_options.bench_level = static_cast<int>(tier);
      chosen.preset = Preset::Auto;
      chosen.priority = Priority::Balance;
    }
    // L'exploration du benchmark (bench.toml) ne vaut que pour cette carte et ce pilote, et pas pour un
    // essai.
    const GpuFacts here = gpu_facts(**screen);
    const CalibrationHost host = calibration_host(here, display_height((*screen)->window()), library.get(),
                                                  library_wanted, *engines, catalog, paths);
    const bool explored = measured_tier && !file_options.bench_level && measured_tier->gpu == here.name &&
                          measured_tier->driver == here.driver;
    chosen = apply_preset(
        chosen,
        Machine{.tier = tier,
                .display_height = display_height((*screen)->window()),
                .display_hz = display_refresh_hz((*screen)->window()),
                .explored_work_height = explored ? measured_tier->work_height : 0,
                .explored_rife_model = explored ? measured_tier->rife_model : std::string(),
                .calibrated = file_options.bench_level ? std::nullopt
                                                       : calibrated_chain(settings_path.parent_path(), host)},
        source);
    PlaySettings settings = settings_for(chosen, file_options);
    if (auto valid = validate(settings); !valid) return std::unexpected(std::move(valid.error()));
    static_cast<void>((*ui)->controls().set_volume(settings.volume));
    // Le module chargé a été remplacé sur le disque (moteur mis à jour depuis l'écran des moteurs) : il ne se
    // recharge pas sous le même nom dans ce processus, le lecteur se relance sur cette vidéo.
    if (library && library->replaced_on_disk() && paths.relaunch != nullptr) {
      log->info("moteur mis à jour : le lecteur se relance");
      *paths.relaunch = true;
      return {};
    }
    // Un autre moteur choisi depuis (accueil, Paramètres) : la bibliothèque se rouvre sur lui, sinon le choix
    // serait enregistré sans être appliqué.
    const std::optional<core::EngineId> wanted =
        engines->preferred[static_cast<std::size_t>(core::Usage::Upscale)];
    if (library && wanted != library_wanted) library.reset();
    if (!library) {
      // Un moteur choisi à l'accueil depuis le lancement, installé par une version précédente : mis à jour
      // avant d'ouvrir, comme au lancement (une seule relance).
      bool stale_updated = false;
      if (wanted != checked_engine) {
        auto stale = update_stale_engines(**screen, **ui, catalog, *engines, stale_updated);
        if (!stale) return std::unexpected(std::move(stale.error()));
        if (!*stale) return {};
      }
      if (stale_updated && paths.relaunch != nullptr) {
        log->info("moteurs mis à jour : le lecteur se relance");
        *paths.relaunch = true;
        return {};
      }
      show_loading(**screen, **ui, i18n::ui_tr("Loading"));
      const gpu::BackendInit probe{
          .render = &(*screen)->device(), .cache_dir = paths.engines, .log = backend_log()};
      auto opened = open_backend(catalog, paths, *engines, probe);
      if (!opened) return std::unexpected(std::move(opened.error()));
      library = std::move(*opened);
      library_wanted = wanted;
    }

    Player player{.options = file_options,
                  .paths = paths,
                  .catalog = catalog,
                  .screen = **screen,
                  .ui = **ui,
                  .library = std::move(library),
                  .video = video,
                  .settings_path = settings_path,
                  .from_file = *saved,
                  .started = settings,
                  .snapshot = std::move(snapshot),
                  .details = settings.stats.all,
                  .next_key = next_key,
                  .engines = std::move(*engines),
                  .display = window_display((*screen)->window())};
    player.shots = &shots;
    {
      // « Me prévenir » : une version plus récente vue par la dernière vérification, annoncée une fois par
      // lancement.
      const UpdateState update = load_update_state(update_file);
      if (!update_announced && saved->update_mode == UpdateMode::Notify && !update.latest_version.empty()) {
        update_announced = true;
        player.notice = i18n::ui_tr_format("Update {} available (Settings > General > Application)",
                                           update.latest_version);
      }
    }
    {
      // P3.11 : une calibration faite sur une autre carte ne sert plus ; la refaire est proposé une fois par
      // lancement.
      const auto calibration = load_calibration(calibration_file(settings_path.parent_path()));
      if (!calibration_told && calibration && !calibration->gpu.empty() && calibration->gpu != here.name) {
        calibration_told = true;
        const i18n::Translated redo =
            i18n::ui_tr("New graphics card: optimise for this PC again (Video > Post-processing)");
        player.notice = player.notice.empty() ? redo : player.notice + i18n::Translated::data(" · ") + redo;
      }
    }
    if (!set_aside_notes.empty()) {
      // Déjà traduites par set_aside() : montrées telles quelles.
      const i18n::Translated notes = i18n::Translated::data(set_aside_notes);
      player.notice = player.notice.empty() ? notes : notes + i18n::Translated::data(" · ") + player.notice;
      set_aside_notes.clear();
    }
    if (player.display) {
      log->info("écran {} : {}", player.display->connector, describe(*player.display));
    } else {
      log->info("écran : EDID illisible, OLED inconnu");
    }
    std::int64_t start_us = forced_start.value_or(file_options.start_us);
    bool start_paused = std::exchange(forced_paused, false);
    const bool forced = forced_start.has_value();
    forced_start.reset();
    // settings.toml ([playback] resume_position) l'autorise, et --no-resume l'écarte pour une lecture.
    if (file_options.resume && player.engines.resume_position) {
      player.resume_path = paths::resume(settings_path.parent_path());
      if (start_us == 0 && !forced) {
        if (const auto at = resume_position(player.resume_path, file)) {
          start_us = *at;
          player.resumed = true;
          log->info("reprise à {}", format_position(start_us));
        }
      }
    }
    std::expected<void, Refusal> result;
    std::optional<std::filesystem::path> next_file;
    while (true) {
      playing.store(true);
      auto outcome = run_chain(player, settings, start_us, start_paused);
      playing.store(false);
      // P3.10 : « Annuler et regarder » pendant une construction : la même vidéo, sans agrandissement ni
      // interpolation, le temps d'une lecture ; play.toml n'en garde rien.
      if (!outcome && outcome.error().key == kWatchNowKey && (settings.upscale || settings.rife_factor > 0)) {
        settings.upscale = false;
        settings.rife_factor = 0;
        settings.rife_auto = false;
        file_options.given.emplace_back("--no-upscale");
        file_options.given.emplace_back("--rife");
        file_options.rife_factor = 0;
        player.notice = i18n::ui_tr("Engine build cancelled: playing without upscaling or smoothing");
        continue;
      }
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
        follow_wizard(*request, **outcome, file);  // A25-01 : réglages de l'assistant relus
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
      // « Refaire le banc » (P3.12) : la chaîne est fermée, sa mémoire vidéo libérée ; la calibration
      // mesure la machine seule, puis la lecture reprend avec ce qu'elle a trouvé.
      if ((*outcome)->calibrate) {
        auto calibrated = calibrate_on_screen(player.screen, player.ui, player.paths,
                                              player.settings_path.parent_path(), settings);
        if (!calibrated) {
          player.quit = true;
          break;
        }
        settings = std::move(*calibrated);
      }
      start_us = (*outcome)->position_us;
      start_paused = (*outcome)->paused;
      player.details = settings.stats.all;
      if ((*outcome)->next_file) {
        save_before_next(player, **outcome, settings, file_options);
        player.unsaved.reset();
        next_file = (*outcome)->next_file;
        forced_start = (*outcome)->next_start_us;
        forced_paused = (*outcome)->next_paused;
        break;
      }
      player.unsaved = settings;  // écrits quand la chaîne rouverte est ouverte (run_chain)
      log->info("réouverture à {}", format_position(start_us));
    }
    // Une vidéo choisie à l'accueil qui échoue y ramène avec la raison, plutôt que de fermer le
    // lecteur ; lancé sur un fichier, l'échec reste celui du programme (code de sortie, message).
    std::string failed;
    if (!result) {
      // Un fichier glissé qui ne s'ouvre pas ramène à l'accueil avec la raison : la lecture en cours n'est
      // pas une raison de quitter.
      // P3.10 : ouvert par double-clic, l'échec ne ferme plus la fenêtre sans un mot ; seuls les lancements
      // pilotés (essais, banc, captures) gardent le code de sortie.
      if (scripted || player.quit) return result;
      log->error("{}", result.error().detail);
      failed = i18n::tr_format("\"{}\" stopped: {}", file.filename().string(), visible_error(result.error()));
      log->error("{}", failed);
    }
    // Ce que la lecture a changé suit dans la vidéo suivante ou l'accueil.
    library = std::move(player.library);
    snapshot = std::move(player.snapshot);
    next_key = player.next_key;
    *engines = std::move(player.engines);
    // Un changement de moteur pendant la lecture a déjà rouvert la bibliothèque sur le préféré.
    if (library && engine_of_library(library->path()) ==
                       engines->preferred[static_cast<std::size_t>(core::Usage::Upscale)])
      library_wanted = engines->preferred[static_cast<std::size_t>(core::Usage::Upscale)];
    if (!failed.empty()) {
      problem = std::move(failed);
      file.clear();
    } else if (next_file) {
      file = std::move(*next_file);  // vide (PlayFile) : l'accueil
    } else if (from_home && !player.quit) {
      file.clear();  // fin de la vidéo : retour à l'accueil
    } else {
      return {};
    }
    if (auto reread = load_settings(settings_path)) {
      saved = std::move(*reread);
      i18n::set_language(i18n::language_from_code(saved->language));
    }
  }
}

// --bake-looks : la pile de LUT des réglages en une seule, comme le lecteur la lit (intensités, ordre).
std::expected<void, Refusal> bake_looks(const std::filesystem::path& settings_file,
                                        const std::filesystem::path& luts_dir,
                                        const std::filesystem::path& out) {
  auto settings = load_settings(settings_file);
  if (!settings) return std::unexpected(std::move(settings.error()));
  const PlaySettings& s = *settings;
  LoadedLooks looks = load_looks(s, luts_dir);
  if (!looks.failed.empty())
    return refuse(RefusalCode::FileUnreadable,
                  std::format("--bake-looks : {} illisible", looks.failed.front()));
  const std::vector<LookLayer> slots = look_slots(s);
  std::array<const render::CubeLut*, render::kLookSlots> luts{};
  std::array<double, render::kLookSlots> mixes{};
  for (std::size_t i = 0; i < render::kLookSlots && i < slots.size(); ++i) {
    const std::optional<render::CubeLut>& lut = looks.luts.at(i);
    if (!lut.has_value()) continue;
    luts.at(i) = &*lut;
    mixes.at(i) = slots[i].mix;
    log_channel("app")->info("--bake-looks : {} à {:.0f} %", slots[i].key, 100.0 * slots[i].mix);
  }
  render::CubeLut baked = render::compose_luts(luts, mixes);
  baked.title = "ajp anime HDR (pile composee)";
  std::ofstream file(out, std::ios::binary);
  file << render::write_cube(baked);
  if (!file)
    return refuse(RefusalCode::FileUnreadable, std::format("--bake-looks : {} ne s'écrit pas", out.string()));
  log_channel("app")->info("--bake-looks : {} écrit ({} points)", out.string(), baked.size);
  return {};
}

}  // namespace ajp::app::play
