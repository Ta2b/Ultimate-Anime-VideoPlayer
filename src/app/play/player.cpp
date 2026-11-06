#include "app/play/player.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
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
#include "app/play/build_progress.h"
#include "app/play/calibration_images.h"
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
#include "app/play/i18n.h"
#include "app/play/input.h"
#include "app/play/look_catalog.h"
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
#include "app/play/updater.h"
#include "app/play/upscale_stage.h"
#include "app/play/visible_error.h"
#include "core/clock.h"
#include "core/log.h"
#include "core/settings.h"
#include "media/demuxer.h"
#include "media/selection.h"
#include "prepare/sha256.h"
#include "prepare/utf8.h"

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
  return i18n::tr_format("{} was invalid: copy kept as {}", path.filename().string(),
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
  std::string unreadable;  // le fichier donné ne se lit pas : dit à l'accueil, qui s'ouvre à sa place
  if (!options.file.empty()) {
    auto video = probe_video(options.file);
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
  // Vérification des mises à jour, en arrière-plan et hors lecture : le fil s'arrête avec le lecteur.
  std::jthread update_check;
  const std::filesystem::path update_file = paths.data / "update.toml";
  const UpdatePaths update_paths{
      .data = paths.data, .install = paths.executable.parent_path(), .executable = paths.executable};
  const std::filesystem::path settings_path =
      (paths.settings.empty() ? paths.engines.parent_path() : paths.settings) / "play.toml";
  std::vector<std::string> startup_notes;  // fichiers mis de côté, dits à l'accueil
  auto saved = load_settings(settings_path);
  if (saved) i18n::set_language(i18n::language_from_code(saved->language));
  if (!saved) {
    log->warn("{} ; réglages par défaut", saved.error().detail);
    startup_notes.push_back(set_aside(settings_path, saved.error()));
    saved = PlaySettings{};
  }
  if (!options.bench_out && !options.data_dir) {
    const std::int64_t now =
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();
    const UpdateState seen = load_update_state(update_file);
    if (update_due(seen, saved->update_mode, saved->update_frequency, now)) {
      update_check = std::jthread(
          [=, mode = saved->update_mode, channel = saved->update_channel](const std::stop_token& stop) {
            UpdateState state = load_update_state(update_file);
            const auto message = run_update_check(update_paths, mode, channel, state, stop);
            if (const auto kept = save_update_state(update_file, state); !kept)
              log_channel("app")->warn("update.toml : {}", kept.error().detail);
            if (message) {
              log_channel("app")->info("mise à jour : {}", *message);
            } else {
              log_channel("app")->info("mise à jour : {}", message.error().detail);
            }
          });
    }
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
  set_no_looks(options.no_looks);
  set_clean_looks(options.clean_looks);            // audit F1 : grain et paliers non étirés par la pile
  set_contract_v2_looks(options.hdr_contract_v2);  // 8.4 : ver9-couleur.cube à la place de la pile par défaut
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
      .kde_sdr_nits = options.hdr_contract_v2 ? std::nullopt : options.kde_sdr_nits,
      .kde_sdr_auto = options.kde_sdr_auto && !options.hdr_contract_v2,
      .contract_v2 = options.hdr_contract_v2,
      .metadata_off = options.hdr_metadata_off,
      .smooth_motion = saved->smooth_motion && saved->rife_factor >= 2,
      .allow_software = options.software_gpu,
      .grade_deband = options.grade_deband,
      .skin_mask = options.skin_mask,
      .grade_ring = options.grade_ring,
      .present = options.present,
      .shader_cache = paths.data / "shaders.cache",
      .screen_stamps = options.stats ? std::filesystem::path(*options.stats).replace_extension(".screen.tsv")
                                     : std::filesystem::path()});
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
  // Un moteur débranché depuis (MIGraphX, OpenVINO) sort des choix : Vulkan prend sa place.
  for (const core::EngineId id : {core::EngineId::MiGraphX, core::EngineId::OpenVino}) {
    if (!chosen(*engines, id)) continue;
    log->info("moteurs : {} débranché de ce lecteur, Vulkan à sa place", engine_label(id));
    *engines = without_engine(std::move(*engines), id);
  }
  // Les moteurs téléchargeables de cette version (engines.lock livré avec le lecteur).
  const EngineCatalog catalog = load_catalog(
      paths.engine_index, paths.engines.parent_path(),
      saved->engines_dir.empty() ? std::filesystem::path() : prepare::path_from_utf8(saved->engines_dir));
  if (!catalog.problem.empty()) log->info("{}", catalog.problem);
  const EngineScreen engine_screen{.screen = **screen,
                                   .ui = **ui,
                                   .options = options,
                                   .paths = paths,
                                   .catalog = catalog,
                                   .engines_path = engines_path};
  // Une vidéo ouverte directement (double-clic, argument) se lit tout de suite : l'assistant ne
  // la bloque pas, et il se présentera au prochain lancement depuis l'accueil.
  if (options.first_run || (!engines->first_run_done && !given)) {
    auto shown = run_engine_screen(engine_screen, *engines, profiles, false, next_key, snapshot);
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
  for (const std::string& note : startup_notes) {
    if (note.empty()) continue;
    problem += problem.empty() ? note : " · " + note;
  }
  std::shared_ptr<BackendLibrary> library;       // ouverte à la première vidéo, gardée pour les suivantes
  std::optional<core::EngineId> library_wanted;  // le moteur préféré quand elle a été ouverte
  // La pré-analyse des plans (chain_shots.h) et son cache : seule la vidéo lue s'analyse.
  ShotWork shots{.cache_dir = paths.data / "plans"};
  if (options.probe_nvdec) shots.probe.cuvid_resize = media::kProbeCuvidSize;  // 8.8
  bool replaced = false;  // le fichier vient d'être glissé sur la fenêtre (ou est l'épisode voisin)
  // PlayFile (la calibration du grade) : l'instant et la pause du fichier suivant, à la place de la reprise.
  std::optional<std::int64_t> forced_start;
  bool forced_paused = false;
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
      auto probed = probe_video(file);
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
    profiles.source =
        core::SourceInfo{.width = video.width, .height = video.height, .fps = fps_of(video.frame_rate)};

    PlaySettings from_profile = *saved;
    // L'essai d'une variante (« ajp compare », la calibration de la machine) joue son play.toml tel quel : le
    // profil de démarrage referait la chaîne.
    const bool variant_trial = options.bench_out && !options.bench_level;
    if (const auto& start = profiles.data.file.profiles[static_cast<std::size_t>(profiles.active - 1)];
        start && !variant_trial) {
      std::vector<std::string> notes;
      from_profile = apply_profile(*start, profiles.source, *saved, notes);
      for (const std::string& line : notes) log->warn("profil {} : {}", profiles.active, line);
      log->info("profil {} « {} »", profiles.active, start->name);
    }
    // La tranche mesurée par « Évaluer ma machine » (bench.toml), si elle est de cette carte.
    std::optional<BenchResult> measured_tier;
    if (auto bench = load_bench(settings_path.parent_path() / "bench.toml"))
      measured_tier = std::move(*bench);
    // Le préréglage (Auto, Rapide…) refait la chaîne pour cette carte, cet écran et cette vidéo ;
    // Personnalisé laisse les réglages tels quels.
    Tier tier = effective_tier(gpu_facts(**screen), measured_tier);
    if (file_options.bench_level) {
      // Un essai du benchmark : la chaîne du niveau demandé (ou de l'estimé), sans priorité ni profil.
      if (*file_options.bench_level >= 0) tier = static_cast<Tier>(*file_options.bench_level);
      file_options.bench_level = static_cast<int>(tier);
      from_profile.preset = Preset::Auto;
      from_profile.priority = Priority::Balance;
    }
    // L'exploration du benchmark (bench.toml) ne vaut que pour cette carte et ce pilote, et pas pour un
    // essai.
    const GpuFacts here = gpu_facts(**screen);
    const bool explored = measured_tier && !file_options.bench_level && measured_tier->gpu == here.name &&
                          measured_tier->driver == here.driver;
    from_profile =
        apply_preset(from_profile,
                     Machine{.tier = tier,
                             .display_height = display_height((*screen)->window()),
                             .display_hz = display_refresh_hz((*screen)->window()),
                             .explored_work_height = explored ? measured_tier->work_height : 0,
                             .explored_rife_model = explored ? measured_tier->rife_model : std::string()},
                     profiles.source);
    PlaySettings settings = settings_for(from_profile, file_options);
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
      show_loading(**screen, **ui, "Chargement");
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
                  .profiles = std::move(profiles),
                  .engines = std::move(*engines),
                  .display = read_display(display_name((*screen)->window()))};
    player.shots = &shots;
    {
      // « Me prévenir » : une version plus récente vue par la dernière vérification, annoncée une fois par
      // lancement.
      static bool announced = false;
      const UpdateState update = load_update_state(update_file);
      if (!announced && saved->update_mode == "notify" && !update.latest_version.empty()) {
        announced = true;
        player.notice =
            i18n::tr_format("Update {} available (Settings > General > Application)", update.latest_version);
      }
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
    // settings.toml ([playback] resume_position) l'autorise, et --no-resume l'écarte pour une lecture ; la
    // vidéo de la calibration du grade ne se reprend ni ne se retient.
    if (file_options.resume && player.engines.resume_position && file != calibration_clip()) {
      player.resume_path = settings_path.parent_path() / "resume.tsv";
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
      if ((!from_home && !replaced) || player.quit) return result;
      log->error("{}", result.error().detail);
      failed = i18n::tr_format("\"{}\" stopped: {}", file.filename().string(), visible_error(result.error()));
      log->error("{}", failed);
    }
    replaced = false;
    // Ce que la lecture a changé suit dans la vidéo suivante ou l'accueil.
    library = std::move(player.library);
    snapshot = std::move(player.snapshot);
    next_key = player.next_key;
    profiles = std::move(player.profiles);
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
      replaced = !file.empty();
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
    if (!looks.luts.at(i)) continue;
    luts.at(i) = &*looks.luts.at(i);
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
