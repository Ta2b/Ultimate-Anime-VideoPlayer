#include "app/play/player.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <variant>

#include <SDL3/SDL_error.h>
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
#include "app/play/crash_dump.h"
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
#include "app/play/player_machine.h"
#include "app/play/player_run.h"
#include "app/play/pumped_task.h"
#include "app/play/relaunch_target.h"
#include "app/play/release_refresh.h"
#include "app/play/resource_usage.h"
#include "app/play/resume.h"
#include "app/play/scripted_input.h"
#include "app/play/session.h"
#include "app/play/ui.h"
#include "app/play/update_thread.h"
#include "app/play/updater.h"
#include "app/play/upscale_stage.h"
#include "app/play/user_keys.h"
#include "app/play/visible_error.h"
#include "core/clock.h"
#include "core/log.h"
#include "core/settings.h"
#include "core/thread.h"
#include "core/utf8_path.h"
#include "media/demuxer.h"
#include "media/selection.h"
#include "media/track.h"
#include "prepare/sha256.h"
#include "prepare/utf8.h"

namespace ajp::app::play {
namespace {

// Un fichier de configuration que le lecteur n'a pas compris sera réécrit au prochain changement
// de réglage : il est mis de côté d'abord (« nom.invalide-AAAAMMJJ-HHMMSS »), pour qu'aucun profil
// ni réglage valide qu'il contenait ne se perde sans copie. Un fichier illisible (verrou, droits) n'est pas
// déplacé : load_settings et save_settings le laissent intact, et la séance le dit (B22-02). Rend le message
// à montrer à l'accueil.
std::string set_aside(const std::filesystem::path& path, const Refusal& why) {
  std::error_code error;
  if (why.code == RefusalCode::FileUnreadable)
    return i18n::tr_format(
        "{} cannot be read: this session starts from the default settings and does not "
        "change the file",
        core::to_utf8(path.filename()));
  if (!std::filesystem::exists(path, error)) return {};
  const auto now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
  std::filesystem::path aside = path;
  aside += std::format(".invalide-{:%Y%m%d-%H%M%S}", now);
  std::filesystem::rename(path, aside, error);
  if (error) {
    log_channel("app")->warn("{} : mise de côté impossible ({})", core::to_utf8(path.filename()),
                             error.message());
    return {};
  }
  log_channel("app")->warn("{} mis de côté dans {}", core::to_utf8(path.filename()),
                           core::to_utf8(aside.filename()));
  return i18n::tr_format("{} was invalid: copy kept as {}", core::to_utf8(path.filename()),
                         core::to_utf8(aside.filename()));
}

// play.toml, avec l'avis des valeurs fautives que le défaut a remplacées (B22-05) ajouté aux notes de départ.
std::expected<PlaySettings, Refusal> load_with_notes(const std::filesystem::path& path,
                                                     std::vector<std::string>& notes) {
  std::vector<std::string> ignored;
  auto loaded = load_settings(path, &ignored);
  if (std::string note = ignored_values_note(ignored); !note.empty()) notes.push_back(std::move(note));
  return loaded;
}

// La piste vidéo que la source choisira, lue d'avance : sa taille fixe le moteur et le ring.
std::expected<media::TrackInfo, Refusal> probe_once(const std::filesystem::path& file) {
  auto demuxer = media::Demuxer::open(file, {.attachments = false});
  if (!demuxer) return std::unexpected(std::move(demuxer.error()));
  const media::TrackChoice choice = media::choose_tracks(demuxer->tracks(), {});
  if (!choice.video) {
    // P3.28 : un fichier audio seul est dit tel quel, pas comme un fichier abîmé.
    Refusal refusal{RefusalCode::TrackUnavailable,
                    std::format("{} : aucune piste vidéo", core::to_utf8(file))};
    if (choice.audio) refusal.key = kAudioOnlyKey;
    return std::unexpected(std::move(refusal));
  }
  const media::TrackInfo& video = *demuxer->track(*choice.video);
  // P3.26 : le profil 5 n'a pas de couche de base HDR10 ou SDR (IPTPQc2) ; les profils 7 et 8 se lisent par
  // leur couche de base, sans les métadonnées dynamiques.
  if (video.dolby_vision_profile == 5) {
    Refusal refusal{RefusalCode::FormatUnsupported,
                    std::format("{} : Dolby Vision profil 5", core::to_utf8(file))};
    refusal.key = kDolbyVision5Key;
    return std::unexpected(std::move(refusal));
  }
  // A15-07 : un format ou une couleur que la chaîne refuserait se dit ici, avant de construire la chaîne.
  if (std::optional<Refusal> unreadable = media::unreadable_video(video)) {
    unreadable->detail = std::format("{} : {}", core::to_utf8(file.filename()), unreadable->detail);
    return std::unexpected(std::move(*unreadable));
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

// A33-02 : SDL coupe l'économiseur dès son ouverture ; l'accueil et l'assistant le laissent agir, la
// lecture le coupe pendant qu'une vidéo avance (sleep_watch.h).
void allow_screensaver() {
  if (!SDL_EnableScreenSaver()) log_channel("app")->warn("économiseur d'écran : {}", SDL_GetError());
}

// Un avis de plus devant ceux de la vidéo qui s'ouvre, une fois : first est vidé.
void announce_first(i18n::Translated& notice, i18n::Translated& first) {
  if (first.empty()) return;
  notice = notice.empty() ? first : first + i18n::Translated::data(" · ") + notice;
  first = {};
}

// Annoncés une fois par lancement, pas à chaque vidéo.
struct Announced {
  bool calibration = false;
};

// « Me prévenir » : une version plus récente vue par la dernière vérification. P3.11 : une calibration faite
// sur une autre carte ne sert plus ; la refaire est proposé.
void announce_once(Player& player, Announced& announced, UpdateNews& news, UpdateMode mode,
                   const std::string& gpu, int display_height) {
  player.update_news = &news;
  const UpdateState update = load_update_state(news.file);
  if (!news.shown.load() && mode == UpdateMode::Notify && !update.latest_version.empty()) {
    news.shown.store(true);
    player.notice =
        i18n::ui_tr_format("Update {} available (Settings > General > Application)", update.latest_version);
  }
  // La calibration de cet écran (A37-10), sinon l'ancien calibration.toml.
  const auto calibration = load_calibration_at(player.settings_path.parent_path(), display_height);
  if (!announced.calibration && calibration && !calibration->gpu.empty() && calibration->gpu != gpu) {
    announced.calibration = true;
    const i18n::Translated redo =
        i18n::ui_tr("New graphics card: calibrate this PC again (Video > Post-processing)");
    player.notice = player.notice.empty() ? redo : player.notice + i18n::Translated::data(" · ") + redo;
  }
}

// A31-02, décision 9, B31-09 : le GPU perdu (TDR, Xid, veille) ne revient pas dans ce processus ; le lecteur
// se relance une fois, en lecture comme à l'accueil ou à l'ouverture des moteurs, et le dit. Une seconde
// perte dans la minute de la relance : le message et l'accueil. L'argument --relaunched le dit même quand
// relaunch.txt n'a pas pu s'écrire (disque plein, B31-07).
struct LossRelaunch {
  const PlayerPaths& paths;
  bool scripted = false;
  bool lost_recently = false;
  std::int64_t launched_s = 0;

  LossRelaunch(const PlayerPaths& at, const PlayOptions& options,
               const std::optional<RelaunchTarget>& reopened, std::int64_t now_s)
      : paths(at),
        scripted(options.stop_after_us > 0 || options.snapshot || !options.keys.empty() ||
                 options.bench_out || options.data_dir),
        lost_recently(options.relaunched == RelaunchedFor::GpuLost ||
                      (reopened && reopened->reason == RelaunchReason::GpuLost)),
        launched_s(now_s) {}

  // Vrai si le lecteur se relance : play() rend alors sans erreur.
  bool operator()(const Refusal& refused, const std::filesystem::path& file, std::int64_t start_us,
                  bool paused) const {
    if (scripted || refused.code != RefusalCode::DeviceLost) return false;
    if (!relaunch_allowed(lost_recently, launched_s)) return false;
    if (!ask_relaunch(paths, file, start_us, paused, RelaunchReason::GpuLost)) return false;
    log_channel("app")->error("{} ; le lecteur se relance", refused.detail);
    return true;
  }
};

// B18-03 : dit dans la langue de play.toml, réglée avant, pas dans celle du module (l'anglais) ; vide si le
// fichier donné se lit.
std::string unreadable_text(const std::filesystem::path& file, const std::optional<Refusal>& why) {
  if (!why) return {};
  return i18n::tr_format("\"{}\" could not be opened: {}", core::to_utf8(file.filename()),
                         visible_error(*why));
}

// Le marqueur est la seule garde contre une boucle de relances : sans lui (disque plein, dossier en
// lecture seule), pas de relance, le lancement continue (Vulkan prend la suite si le module reste périmé).
bool write_relaunch_marker(const std::filesystem::path& marker_path) {
  std::ofstream marker(marker_path);
  marker << "1\n";
  marker.flush();
  if (marker) return true;
  log_channel("app")->warn("moteurs mis à jour : marqueur {} non écrit, pas de relance",
                           core::to_utf8(marker_path.filename()));
  return false;
}

// La position de reprise (resume_from) n'est lue que plus bas : la relance la calcule ici, sinon la vidéo
// rouverte repartait de 0:00 (la cible de relance vaut toujours « forcée »).
std::int64_t relaunch_position(const std::optional<std::int64_t>& forced_start,
                               const PlayOptions& file_options, const core::Settings& engines,
                               const std::filesystem::path& settings_path,
                               const std::filesystem::path& file) {
  if (forced_start) return *forced_start;
  if (file_options.resume && engines.resume_position) {
    if (const auto at = resume_from(paths::resume(settings_path.parent_path()), file)) return *at;
  }
  return file_options.start_us;
}

// Le fichier donné est lu avant d'ouvrir la fenêtre : un fichier illisible n'en ouvre aucune.
std::expected<void, Refusal> probe_given_file(const PlayOptions& options,
                                              std::optional<media::TrackInfo>& given,
                                              std::optional<Refusal>& unreadable_why) {
  if (options.file.empty()) return {};
  auto video = probe_video(options.file, false);
  if (video) {
    given = std::move(*video);
  } else if (options.hidden || options.bench_out || options.snapshot || !options.keys.empty()) {
    return std::unexpected(std::move(video.error()));  // un essai scripté : le code de sortie le dit
  } else {
    // Lancé depuis le gestionnaire de fichiers (« Ouvrir avec »), sans terminal : un refus silencieux
    // ne dirait rien. L'accueil s'ouvre avec la raison.
    log_channel("app")->warn("{}", video.error().detail);
    unreadable_why = std::move(video.error());
  }
  return {};
}

}  // namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity,readability-function-size)
std::expected<void, Refusal> play(const PlayOptions& launch_options, const PlayerPaths& paths) {
  const auto log = log_channel("app");
  // Une relance du lecteur (perte du GPU, moteur mis à jour) rouvre la vidéo qui jouait, à son instant
  // (relaunch_target.h, A31-02, A29-04), à la place du fichier des arguments.
  PlayOptions options = launch_options;
  const std::int64_t launched_s = relaunch_now_s();
  const std::optional<RelaunchTarget> reopened_after = take_relaunch(options, paths, launched_s);
  set_bench_child(options.bench_out.has_value());  // l'essai d'un banc parle à son parent (bench_events.h)
  FontSearch fonts = FontSearch::start();  // pendant la lecture des réglages et l'ouverture de l'écran
  std::optional<media::TrackInfo> given;
  std::optional<Refusal> unreadable_why;  // le fichier donné ne se lit pas : dit à l'accueil, à sa place
  if (auto probed = probe_given_file(options, given, unreadable_why); !probed)
    return std::unexpected(std::move(probed.error()));
  std::atomic<bool> playing{false};  // la vérification des mises à jour attend la fin de la vidéo
  UpdateNews update_news{.file = paths::update_state(paths.data)};  // avant le fil qui le pose
  std::jthread update_check;
  const std::filesystem::path settings_path = paths::play_settings(paths::settings_dir(paths));
  std::vector<std::string> startup_notes;  // fichiers mis de côté, dits à l'accueil
  auto saved = load_with_notes(settings_path, startup_notes);
  i18n::set_language(i18n::language_from_code(saved ? saved->language : default_interface_language()));
  if (!saved) {
    log->warn("{} ; réglages par défaut", saved.error().detail);
    startup_notes.push_back(set_aside(settings_path, saved.error()));
    saved = PlaySettings{};
  }
  const std::string unreadable = unreadable_text(options.file, unreadable_why);  // B18-03
  ReleaseRefresh release_index = start_release_refresh(options, paths, *saved);  // A18-02
  const std::jthread resource_log = start_resource_log();                        // A33-08
  load_user_keys(paths::user_keys(settings_path.parent_path()));                 // [keys] seul
  const std::string title = given ? std::format("ajp — {}", core::to_utf8(options.file.filename())) : "ajp";
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
  allow_screensaver();
  auto ui = PlayerUi::create((*screen)->window(), given ? core::to_utf8(options.file.filename()) : "", fonts);
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
  auto engines = load_engine_settings(engines_path, &engines_skipped);
  for (const std::string& skipped : engines_skipped) log->warn("settings.toml : {} écarté", skipped);
  if (!engines) {
    log->warn("{} ; écran de bienvenue", engines.error().detail);
    startup_notes.push_back(set_aside(engines_path, engines.error()));
    engines = core::Settings{};
  }
  // Un moteur retiré (MIGraphX, OpenVINO) est écarté dès la lecture de settings.toml (core::retired_engine).
  const EngineCatalog catalog = engine_catalog(paths, *saved, release_index);
  const EngineScreen engine_screen{.screen = **screen,
                                   .ui = **ui,
                                   .options = options,
                                   .paths = paths,
                                   .catalog = catalog,
                                   .engines_path = engines_path};
  // L'assistant s'ouvre au premier lancement dans tous les cas, double-clic compris (P3.1) : la vidéo
  // demandée attend, puis se lance à la fin. Seuls les lancements pilotés (essais, captures, banc, dossier de
  // réglages à part) le sautent.
  const LossRelaunch relaunch_after_loss(paths, options, reopened_after, launched_s);
  const bool scripted = relaunch_after_loss.scripted;
  if (options.first_run || (!engines->first_run_done && !scripted)) {
    auto shown = run_engine_screen(engine_screen, *engines, false, next_key, snapshot);
    if (!shown && relaunch_after_loss(shown.error(), options.file, 0, false)) return {};
    if (!shown) return std::unexpected(std::move(shown.error()));
    if (!*shown) return {};
    // L'assistant vient d'écrire play.toml (HDR, chaîne calibrée, auto Rapide) : l'accueil part de là, sans
    // quoi il reprendrait les réglages d'avant et les réécrirait par-dessus.
    if (auto reread = load_settings(settings_path)) saved = std::move(*reread);
  }
  // Après l'assistant, qui choisit le mode des mises à jour (A18-03).
  if (!options.bench_out && !options.data_dir)
    update_check = start_update_check(settings_path, paths, playing, update_news);
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
  // Le module remplacé ne se charge pas sous le même nom dans ce processus : le lecteur repart, et le moteur
  // choisi s'ouvre au lancement suivant sans autre geste.
  if (engines_updated && !relaunched && paths.relaunch != nullptr && write_relaunch_marker(relaunch_marker)) {
    log->info("moteurs mis à jour : le lecteur se relance");
    *paths.relaunch = true;
    return {};
  }

  // Lancé sans fichier, l'accueil choisit la vidéo, et il revient à la fin de chacune. Une vidéo
  // glissée sur la fenêtre pendant la lecture prend la place de celle en cours.
  const bool from_home = !given || (reopened_after && launch_options.file.empty());
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
  ShotWork shots = shot_work(paths::shot_cache(paths), options);
  // P3.10, décisions 6 et 10 : un étage refusé ou annulé (« Annuler et regarder ») laisse la vidéo jouer sans
  // lui ; le refus vaut pour la séance, l'annulation pour sa vidéo (stage_fallback.h). play.toml n'en garde
  // rien.
  StageMemory stage_memory;
  ManualTracks manual_tracks;        // B29-11 : d'un épisode au suivant
  i18n::Translated engine_fallback;  // le moteur choisi remplacé à l'ouverture : à dire sur la vidéo (A31-03)
  // PlayFile : l'instant et la pause du fichier suivant, à la place de la reprise.
  std::optional<std::int64_t> forced_start;
  bool forced_paused = false;
  i18n::Translated relaunch_note;  // dit sur la vidéo rouverte après une perte du GPU
  if (reopened_after && given) {
    forced_start = reopened_after->start_us;
    forced_paused = reopened_after->paused;
    if (reopened_after->reason == RelaunchReason::GpuLost)
      relaunch_note = i18n::ui_tr("The graphics card stopped responding: the player restarted");
  }
  Announced announced;  // la mise à jour vue, la calibration d'une autre carte
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
      if (!picked && relaunch_after_loss(picked.error(), {}, 0, false)) return {};
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
        problem = i18n::tr_format("\"{}\" could not be opened: {}", core::to_utf8(file.filename()),
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
      SDL_SetWindowTitle((*screen)->window(),
                         std::format("ajp — {}", core::to_utf8(file.filename())).c_str());
      (*ui)->set_title(core::to_utf8(file.filename()));
    }
    note_crash_stage(std::format("ouverture de {}", core::to_utf8(file.filename())));  // B39-10
    PlayOptions file_options = options;
    file_options.file = file;
    const core::SourceInfo source{
        .width = video.width, .height = video.height, .fps = fps_of(video.frame_rate)};

    PlaySettings chosen = *saved;
    // La tranche mesurée par « Calibrer cette machine » (bench.toml), si elle est de cette carte.
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
    // L'exploration du benchmark (bench.toml) ne vaut que pour cette carte et cette branche de pilote
    // (A37-09), et pas pour un essai.
    const GpuFacts here = gpu_facts(**screen);
    const CalibrationHost host = calibration_host(here, display_height((*screen)->window()), library.get(),
                                                  library_wanted, *engines, catalog, paths);
    const BenchResult* explored =
        measured_tier && !file_options.bench_level && measured_here(*measured_tier, here) ? &*measured_tier
                                                                                          : nullptr;
    const auto machine_now = [&] {
      return machine_of(
          (*screen)->window(), tier, explored,
          file_options.bench_level ? std::nullopt : calibrated_chain(settings_path.parent_path(), host));
    };
    chosen = apply_preset(chosen, machine_now(), source);
    PlaySettings settings = settings_for(chosen, file_options);
    if (auto valid = validate(settings); !valid) return std::unexpected(std::move(valid.error()));
    static_cast<void>((*ui)->controls().set_volume(settings.volume));
    // Le module chargé a été remplacé sur le disque (moteur mis à jour depuis l'écran des moteurs) : il ne se
    // recharge pas sous le même nom dans ce processus, le lecteur se relance sur cette vidéo.
    const auto relaunch_start_us = [&] {
      return relaunch_position(forced_start, file_options, *engines, settings_path, file);
    };
    if (library && library->replaced_on_disk() && paths.relaunch != nullptr) {
      log->info("moteur mis à jour : le lecteur se relance");
      ask_relaunch(paths, file, relaunch_start_us(), forced_paused, RelaunchReason::EngineUpdated);
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
        ask_relaunch(paths, file, relaunch_start_us(), forced_paused, RelaunchReason::EngineUpdated);
        return {};
      }
      show_loading(**screen, **ui, i18n::ui_tr("Loading"));
      const gpu::BackendInit probe{
          .render = &(*screen)->device(), .cache_dir = paths.engines, .log = backend_log()};
      auto opened = open_backend(catalog, paths, *engines, probe);
      if (!opened) {
        if (opened.error().code == RefusalCode::Cancelled) return {};  // fenêtre fermée (B30-06)
        if (relaunch_after_loss(opened.error(), file, relaunch_start_us(), forced_paused)) return {};
        if (scripted) return std::unexpected(std::move(opened.error()));
        // A18-13, B31-09 : la vidéo, choisie à l'accueil ou par double-clic, y ramène avec la raison.
        problem = i18n::tr_format("\"{}\" could not be opened: {}", core::to_utf8(file.filename()),
                                  visible_error(opened.error()));
        file.clear();
        continue;
      }
      library = std::move(*opened);
      library_wanted = wanted;
      // A31-03 : le moteur choisi refuse ce GPU (pilote trop ancien, module abîmé) : un autre le remplace,
      // plus lent ; dit à l'écran une fois, la raison au journal et dans Général > Moteurs.
      if (const std::optional<core::EngineId> in_use = engine_of_library(library->path());
          wanted && in_use && wanted != in_use)
        engine_fallback =
            i18n::ui_tr_format("{} unavailable: {} is used, slower (Settings > General > Engines)",
                               engine_label(*wanted), engine_label(*in_use));
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
    player.shots = options.bench_out ? nullptr : &shots;  // A42-05 (chain_shots.h, shot_work)
    player.stages = &stage_memory;
    player.manual_tracks = &manual_tracks;
    player.relaunch_on_loss =
        !scripted && relaunch_allowed(relaunch_after_loss.lost_recently, relaunch_after_loss.launched_s);
    announce_once(player, announced, update_news, saved->update_mode, here.name, host.display_height);
    announce_first(player.notice, relaunch_note);
    announce_first(player.notice, engine_fallback);
    // Déjà traduites par set_aside() : montrées telles quelles.
    i18n::Translated aside = i18n::Translated::data(std::exchange(set_aside_notes, {}));
    announce_first(player.notice, aside);
    if (player.display) {
      log->info("écran {} : {}", player.display->connector, describe_for_log(*player.display));
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
        if (const auto at = resume_from(player.resume_path, file)) {
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
          // Un moteur mis à jour relance le lecteur : sur cette vidéo, à cet instant (A29-04).
          if (paths.relaunch != nullptr && *paths.relaunch)
            ask_relaunch(paths, file, (*outcome)->position_us, (*outcome)->paused,
                         RelaunchReason::EngineUpdated);
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
      if ((*outcome)->rederive)
        settings = rederived(settings, machine_now(), source, file_options, player.kept);
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
      if (result.error().key == kWindowClosedKey) return {};  // la croix pendant un chargement : on quitte
      // Le GPU perdu : une relance sur cette vidéo, à cet instant (relaunch_after_loss).
      if (relaunch_after_loss(result.error(), file, player.stopped_at_us, false)) return {};
      log->error("{}", result.error().detail);
      failed = i18n::tr_format("\"{}\" stopped: {}", core::to_utf8(file.filename()),
                               visible_error(result.error()));
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
    const render::CubeLut* lut = looks.luts.at(i).get();
    if (lut == nullptr) continue;
    luts.at(i) = lut;
    mixes.at(i) = slots[i].mix;
    log_channel("app")->info("--bake-looks : {} à {:.0f} %", slots[i].key, 100.0 * slots[i].mix);
  }
  render::CubeLut baked = render::compose_luts(luts, mixes);
  baked.title = "ajp anime HDR (pile composee)";
  std::ofstream file(out, std::ios::binary);
  file << render::write_cube(baked);
  if (!file)
    return refuse(RefusalCode::FileUnreadable,
                  std::format("--bake-looks : {} ne s'écrit pas", core::to_utf8(out)));
  log_channel("app")->info("--bake-looks : {} écrit ({} points)", core::to_utf8(out.filename()), baked.size);
  return {};
}

}  // namespace ajp::app::play
