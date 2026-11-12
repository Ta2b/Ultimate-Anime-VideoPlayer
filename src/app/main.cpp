// ajp : « ajp play <fichier> [options] », ou « ajp <fichier> » ; « ajp --register » l'inscrit
// comme lecteur vidéo auprès du système (default_player.h).
//
// Tranche verticale : la ligne de commande complète (export, bench, engines…) vient du
// chantier app-shell ; ici, seulement la lecture.
#include <SDL3/SDL_filesystem.h>
#include <SDL3/SDL_init.h>
#include <SDL3/SDL_messagebox.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

extern "C" {
#include <libavutil/log.h>
}

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <tuple>
#ifdef _WIN32
#include <io.h>
#include <process.h>
#else
#include <spawn.h>
#include <unistd.h>

extern char** environ;  // NOLINT : posix_spawn le prend tel quel
#endif
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "app/play/auto_bench_run.h"
#include "app/play/crash_dump.h"
#include "app/play/data_dirs.h"
#include "app/play/default_player.h"
#include "app/play/i18n.h"
#include "app/play/latest_writer.h"
#include "app/play/launch_data.h"
#include "app/play/machine_calibration.h"
#include "app/play/options.h"
#include "app/play/paths.h"
#include "app/play/player.h"
#include "app/play/problem_report.h"
#include "app/play/pumped_task.h"
#include "app/play/relaunch_args.h"
#include "app/play/relaunch_target.h"
#include "app/play/settings.h"
#include "app/play/single_instance.h"
#include "app/play/updater.h"
#include "app/play/visible_error.h"
#include "core/log.h"
#include "core/log_file.h"
#include "core/log_redact.h"
#include "core/thread.h"
#include "core/thread_role.h"
#include "core/utf8_path.h"
#include "media/ffmpeg_log.h"
#include "prepare/utf8.h"

#ifdef _WIN32
// timeapi.h veut windows.h avant lui.
// clang-format off
#include <windows.h>
#include <timeapi.h>
// clang-format on
#endif

#ifdef _WIN32
// Portables à deux GPU (Optimus, PowerXpress) : les pilotes NVIDIA et AMD donnent le GPU dédié,
// à pleine puissance, à l'exécutable qui exporte ces symboles (RTX 4050 mobile, RX 6500M) plutôt
// que l'iGPU qui pilote l'écran.
extern "C" {
__declspec(dllexport) unsigned long NvOptimusEnablement = 1;         // NOLINT
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;  // NOLINT
}
#endif

namespace {

// Avant SDL_Init : les erreurs de FFmpeg vont au journal (ses avertissements de sonde sont du bruit), et SDL
// reçoit le nom, la version et l'identifiant de l'application (B19-01). L'identifiant devient l'app_id
// Wayland et le WM_CLASS X11 : le bureau rattache la fenêtre à la fiche .desktop (icône, épinglage).
void prepare_libraries() {
  ajp::media::route_ffmpeg_log_to_journal();
  SDL_SetAppMetadata(std::string(ajp::app::play::kAppName).c_str(),
                     std::string(ajp::app::play::installed_version()).c_str(),
                     std::string(ajp::app::play::kAppId).c_str());
}

// SDL rend ses chemins en UTF-8, quel que soit le système.
std::filesystem::path sdl_path(const char* path) {
  return path != nullptr ? ajp::prepare::path_from_utf8(path) : std::filesystem::path();
}

// Le lecteur, que l'inscription nomme : à côté de SDL_GetBasePath hors de macOS, où AJP.app
// s'inscrit par son Info.plist.
#ifdef _WIN32
constexpr std::string_view kExecutable = "ajp.exe";
#else
constexpr std::string_view kExecutable = "ajp";
#endif

// Les modules des backends (bibliothèques MODULE de CMake : .so hors de Windows, macOS compris).
#ifdef _WIN32
constexpr std::string_view kModuleSuffix = ".dll";
#else
constexpr std::string_view kModuleSuffix = ".so";
#endif

// Les dossiers où chercher un module : celui de l'exécutable, puis, dans un paquet .app (où
// SDL_GetBasePath rend Contents/Resources), Contents/PlugIns.
std::vector<std::filesystem::path> module_dirs(const std::filesystem::path& base) {
  std::vector<std::filesystem::path> dirs{base};
#ifdef __APPLE__
  dirs.push_back(base / ".." / "PlugIns");
#endif
  return dirs;
}

constexpr int kKeptLogs = 3;  // ajp.log.1 à ajp.log.3

// Le journal est ouvert : un refus s'y écrit (A18-10).
bool& log_open() {
  static bool open = false;
  return open;
}

int fail(const ajp::Refusal& refusal) {
  if (log_open()) {
    ajp::log_channel("app")->error("{} {}", ajp::name(refusal.code), refusal.detail);  // stderr compris
  } else {
    std::fprintf(stderr, "ajp : %s %s\n", std::string(ajp::name(refusal.code)).c_str(),
                 refusal.detail.c_str());
  }
  return 1;
}

// Un refus avant toute fenêtre, lancé par un double-clic : sans console, personne ne lit stderr, une boîte le
// dit (A18-07, A18-10). SDL_ShowSimpleMessageBox se passe de SDL_Init. Pas de boîte pendant un essai scripté
// (--keys, --snapshot, banc, fenêtre cachée), qui doit finir seul.
int fail_shown(const ajp::Refusal& refusal, const ajp::app::play::PlayOptions& options) {
  const int code = fail(refusal);
  if (!options.keys.empty() || options.snapshot || options.bench_out || options.hidden) return code;
  const std::string text = "ajp : " + refusal.detail;
  SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "ajp", text.c_str(), nullptr);
  return code;
}

// C18-04 : une ligne de commande refusée (option inconnue ou retirée, raccourci ou fiche .desktop d'une
// version d'avant) se dit aussi dans une boîte quand personne ne lit stderr : ni terminal, ni essai scripté.
int fail_arguments(const ajp::Refusal& refusal, std::span<const std::string_view> args) {
  const int code = fail(refusal);
  const bool scripted = std::ranges::any_of(args, [](std::string_view arg) {
    return arg.starts_with("--hidden") || arg.starts_with("--keys") || arg.starts_with("--snapshot") ||
           arg.starts_with("--bench-out");
  });
#ifdef _WIN32
  const bool terminal = _isatty(_fileno(stderr)) != 0;
#else
  const bool terminal = isatty(STDERR_FILENO) != 0;
#endif
  if (scripted || terminal) return code;
  const std::string text =
      ajp::app::play::i18n::tr_format("The player could not start with these options: {}", refusal.detail);
  SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "ajp", text.c_str(), nullptr);
  return code;
}

// --report=FICHIER de bench et compare : un dossier absent ou protégé rend 1 avec le chemin (A18-11).
int write_report(const std::optional<std::filesystem::path>& file, const std::string& report) {
  if (!file) return 0;
  std::ofstream out(*file, std::ios::binary);
  out << report;
  out.flush();
  if (out) return 0;
  return fail({ajp::RefusalCode::FileUnwritable,
               std::format("--report : {} ne s'écrit pas", ajp::prepare::utf8_path(*file))});
}

// Le diagnostic d'une séance : un plantage laisse un minidump à côté du journal sous Windows (P2.2), et le
// rapport de « Signaler un problème » garde en tête la version, le système et le journal à lire. La marque
// de séance, restée ouverte, dit que la séance d'avant s'est arrêtée brutalement : son journal est alors
// ajp.log.1 (A39-01).
// B39-01 : après une relance voulue (perte du GPU, moteur mis à jour), la séance d'avant n'a pas planté mais
// son journal, ajp.log.1, porte la cause : le rapport le joint aussi.
void start_diagnostics(const std::filesystem::path& settings_dir, const std::filesystem::path& log_file,
                       ajp::app::play::RelaunchedFor relaunched) {
  namespace play = ajp::app::play;
  if (!log_file.empty()) play::install_crash_dumps(log_file.parent_path() / "crashes");  // B36-08
  play::set_report_log_file(log_file);
  play::note_report_fact("ajp", play::installed_version());
  play::note_report_fact("Système", play::os_description());
  if (const std::string build = play::build_id_description(); !build.empty())
    play::note_report_fact("Build", build);
  if (!settings_dir.empty() && play::begin_session(settings_dir))
    ajp::log_channel("app")->warn(
        "la séance précédente s'est arrêtée brutalement : son journal est ajp.log.1");
  if (relaunched != play::RelaunchedFor::None) {
    play::note_previous_session_relaunched(play::relaunched_name(relaunched));
    ajp::log_channel("app")->info("lecteur relancé ({}) : la séance d'avant est dans ajp.log.1",
                                  play::relaunched_name(relaunched));
  }
}

// Les échecs se disent dans une boîte, sauf pendant un essai scripté (--keys, --snapshot, banc, fenêtre
// cachée), qui doit finir seul. Posé dès que les options sont lues.
bool& quiet_failures() {
  static bool quiet = false;
  return quiet;
}

// Une ligne ajoutée à ajp.log quand le journal est déjà fermé (la relance l'a vidé avant execv).
void append_to_log(const std::filesystem::path& log_file, std::string_view line) {
  if (log_file.empty()) return;
  std::ofstream out(log_file, std::ios::binary | std::ios::app);
  out << "[error] " << line << '\n';
}

// B18-01 : une relance ratée ne ferme plus la fenêtre sans un mot. Le journal est déjà fermé : la cause va
// au bout d'ajp.log, sur stderr, et dans une boîte.
int relaunch_failed(const std::filesystem::path& log_file, const std::string& cause) {
  const std::string line = "relance impossible : " + cause;  // i18n-ok : journal
  append_to_log(log_file, line);
  std::fprintf(stderr, "ajp : %s\n", line.c_str());
  if (!quiet_failures()) {
    const std::string text = ajp::app::play::i18n::tr("The player could not restart: open it again.");
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "ajp", text.c_str(), nullptr);
  }
  return 1;
}

// Relance le lecteur avec les mêmes arguments, après la mise à jour d'un moteur ou la perte du GPU : il
// remplace ce processus (POSIX) ou démarre à sa place (Windows). fresh (POSIX) : un processus neuf par
// posix_spawn, l'appelant sortant ensuite ; une fermeture figée ne peut pas faire execv, qui attendrait la
// fin d'un fil bloqué dans un ioctl du pilote (C31-03).
int relaunch(const std::filesystem::path& executable, int argc, char** argv,
             ajp::app::play::RelaunchedFor reason, const std::filesystem::path& log_file,
             bool fresh = false) {
  const std::vector<std::string_view> launched(argv + 1, argv + argc);
  const std::vector<std::string> kept = ajp::app::play::relaunch_arguments(launched, reason);
  ajp::log_channel("app")->info("relance ({})", ajp::app::play::relaunched_name(reason));
  ajp::app::play::end_session();  // une relance voulue n'est pas un plantage
  ajp::close_log();               // execv remplace le processus : la file du journal s'écrit avant
#ifdef _WIN32
  static_cast<void>(fresh);  // _spawnv lance toujours un processus neuf
  // L'exécutable est gardé dans une chaîne qui vit jusqu'au lancement (c_str() d'un temporaire pendrait).
  const std::string program = executable.string();
  std::vector<std::string> arguments{ajp::app::play::windows_quoted(program)};
  for (const std::string& argument : kept) arguments.push_back(ajp::app::play::windows_quoted(argument));
  std::vector<const char*> pointers;
  for (const std::string& argument : arguments) pointers.push_back(argument.c_str());
  pointers.push_back(nullptr);
  if (_spawnv(_P_NOWAIT, program.c_str(), pointers.data()) != -1) return 0;
  const int error = errno;
  return relaunch_failed(log_file, std::format("_spawnv : {} (errno {})", std::strerror(error), error));
#else
  ajp::app::play::close_on_exec_all();
  std::vector<char*> arguments{const_cast<char*>(executable.c_str())};
  for (const std::string& argument : kept) arguments.push_back(const_cast<char*>(argument.c_str()));
  arguments.push_back(nullptr);
  if (fresh) {
    pid_t child = 0;
    const int spawned = posix_spawn(&child, executable.c_str(), nullptr, nullptr, arguments.data(), environ);
    if (spawned == 0) return 0;
    return relaunch_failed(log_file,
                           std::format("posix_spawn : {} (errno {})", std::strerror(spawned), spawned));
  }
  execv(executable.c_str(), arguments.data());
  const int error = errno;  // execv ne revient que sur un échec
  return relaunch_failed(log_file, std::format("execv : {} (errno {})", std::strerror(error), error));
#endif
}

// B31-04 : la relance qu'une fermeture bloquée sur le GPU fait elle-même (relaunch_target.h : exit_stuck),
// avec les arguments du lancement. Posés une fois par run_player, lus par le garde sur son thread.
struct StuckRelaunch {
  std::filesystem::path executable;
  int argc = 0;
  char** argv = nullptr;
  std::filesystem::path log_file;
};
StuckRelaunch& stuck_relaunch_args() {
  static StuckRelaunch args;
  return args;
}
void relaunch_when_stuck() {
  const StuckRelaunch& args = stuck_relaunch_args();
  // Une fermeture bloquée vient d'un GPU qui ne répond plus : la relance se dit perte du GPU (B39-01).
  if (args.argv != nullptr)
    (void)relaunch(args.executable, args.argc, args.argv, ajp::app::play::RelaunchedFor::GpuLost,
                   args.log_file, /*fresh=*/true);
}
// Le dossier des moteurs, et la relance armée avant la lecture.
void before_play(const std::filesystem::path& engines, const std::filesystem::path& executable, int argc,
                 char** argv, const std::filesystem::path& log_file) {
  std::error_code error;
  std::filesystem::create_directories(engines, error);
  stuck_relaunch_args() =
      StuckRelaunch{.executable = executable, .argc = argc, .argv = argv, .log_file = log_file};
  ajp::app::play::set_stuck_relaunch(&relaunch_when_stuck);
}

// « ajp compare <fichier> --a=… --b=… » : deux chaînes côte à côte (auto_bench_run.h).
int run_compare_command(const std::vector<std::string_view>& args) {
  namespace play = ajp::app::play;
  const auto options = play::parse_compare_options(args);
  if (!options) {
    std::fputs(play::compare_usage().c_str(), stderr);
    return fail(options.error());
  }
  if (options->help) {
    std::fputs(play::compare_usage().c_str(), stdout);
    return 0;
  }
  std::error_code error;
  const auto work = std::filesystem::temp_directory_path(error) / "ajp-compare";
  std::filesystem::create_directories(work, error);
  char* compare_pref = SDL_GetPrefPath("ajp", "ajp");
  const auto compare_data = sdl_path(compare_pref);
  SDL_free(compare_pref);
  const play::BenchPlan plan{.executable = sdl_path(SDL_GetBasePath()) / std::string(kExecutable),
                             .video = options->file,
                             .work_dir = work,
                             .data = compare_data,
                             .seconds = options->seconds,
                             .fullscreen = options->fullscreen};
  const play::Variant a{.name = "A", .args = options->a, .settings = options->a_settings};
  const play::Variant b{.name = "B", .args = options->b, .settings = options->b_settings};
  const auto outcome = play::run_compare(plan, a, b, std::stop_token{}, [](const std::string& step) {
    std::fprintf(stdout, "%s\n", step.c_str());
    std::fflush(stdout);
  });
  std::filesystem::remove_all(work, error);
  if (!outcome) return fail(outcome.error());
  const std::string report = play::format_compare_report(*outcome);
  std::fputs(report.c_str(), stdout);
  return write_report(options->report, report);
}

// « ajp bench <fichier> » : les essais sont des processus « ajp play » ; celui-ci n'ouvre ni fenêtre ni SDL.
int run_bench_command(const std::vector<std::string_view>& args) {
  namespace play = ajp::app::play;
  const auto options = play::parse_bench_options(args);
  if (!options) {
    std::fputs(play::bench_usage().c_str(), stderr);
    return fail(options.error());
  }
  if (options->help) {
    std::fputs(play::bench_usage().c_str(), stdout);
    return 0;
  }
  const auto base = sdl_path(SDL_GetBasePath());
  char* pref = SDL_GetPrefPath("ajp", "ajp");
  const auto data = sdl_path(pref);
  SDL_free(pref);
  std::error_code error;
  const auto work = std::filesystem::temp_directory_path(error) / "ajp-bench";
  std::filesystem::create_directories(work, error);
  const play::BenchPlan plan{.executable = base / std::string(kExecutable),
                             .video = options->file,
                             .work_dir = work,
                             .data = data,
                             .seconds = options->seconds,
                             .fullscreen = options->fullscreen,
                             .silent = options->silent};
  const auto report = play::evaluate_machine(plan, data, std::stop_token{}, [](const std::string& step) {
    std::fprintf(stdout, "%s\n", step.c_str());
    std::fflush(stdout);
  });
  std::filesystem::remove_all(work, error);
  if (!report) return fail(report.error());
  std::fputs(report->report.c_str(), stdout);
  return write_report(options->report, report->report);
}

// « ajp calibrate [--display-height=N] [--tier=N] [--visible] » : le bench de la page « Résultats du bench »,
// en ligne de commande (essais sur un poste de mesure) ; écrit calibration.toml dans les données de
// l'utilisateur. --tier : le niveau de la carte (0 Économe à 4 Ultra), qui donne les modèles joués
// (auto_config.h) ; cette commande n'ouvre pas la carte et ne peut pas l'estimer : sans --tier, 3 (Rapide,
// Tier::Fast), le niveau de la RTX 4070 Ti du poste de mesure.
int run_calibrate_command(const std::vector<std::string_view>& args) {
  namespace play = ajp::app::play;
  int display_height = 2160;
  int tier = static_cast<int>(play::Tier::Fast);
  bool silent = true;
  constexpr const char* kUsage = "ajp calibrate [--display-height=N] [--tier=N] [--visible]\n";
  for (const std::string_view arg : args) {
    if (arg.starts_with("--tier=")) {
      const std::string_view text = arg.substr(7);
      const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), tier);
      if (error != std::errc{} || end != text.data() + text.size() || tier < 0 || tier >= play::kTierCount) {
        std::fputs("ajp calibrate : --tier=N, N de 0 (Économe) à 4 (Ultra)\n", stderr);
        return 2;
      }
    } else if (arg.starts_with("--display-height=")) {
      // from_chars et une plage (P2.35) : « abc » ou « 99999 » ne donnent plus 0 ou une hauteur absurde.
      const std::string_view text = arg.substr(17);
      const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), display_height);
      if (error != std::errc{} || end != text.data() + text.size() || display_height < 360 ||
          display_height > 4320) {
        std::fputs("ajp calibrate : --display-height=N, N de 360 à 4320\n", stderr);
        return 2;
      }
    } else if (arg == "--visible") {
      silent = false;
    } else {
      std::fputs(kUsage, stderr);
      return 2;
    }
  }
  const auto base = sdl_path(SDL_GetBasePath());
  char* pref = SDL_GetPrefPath("ajp", "ajp");
  // Le dossier des réglages du lecteur (settings.toml, moteurs choisis), comme le panneau du bench.
  const auto data = sdl_path(pref);
  SDL_free(pref);
  std::error_code error;
  const play::BenchPlan plan{.executable = base / std::string(kExecutable),
                             .video = data / std::string(play::kBenchClipName),
                             .work_dir = play::calibration_work_dir(data),
                             .data = data,
                             .silent = silent,
                             .display_width = display_height * 16 / 9,
                             .display_height = display_height};
  const auto base_settings = play::load_settings(data / "play.toml");
  const auto done =
      play::calibrate_machine(plan, base_settings.value_or(play::PlaySettings{}), display_height,
                              static_cast<play::Tier>(tier), std::stop_token{}, [](float fraction) {
                                std::fprintf(stdout, "%.0f %%\n", 100.0F * fraction);
                                std::fflush(stdout);
                              });
  if (!done) return fail(done.error());
  std::fprintf(
      stdout, "SPAN %s FP16, RIFE %s\n",
      done->models.span.empty() ? std::string(play::kDefaultModelFile).c_str() : done->models.span.c_str(),
      done->models.rife.c_str());
  for (const play::CandidateTrial& t : done->tried)
    std::fprintf(stdout, "%-48s %s  %.1f / %.0f i/s, 1 %% low %.1f, %.1f %% perdues%s\n",
                 play::describe(t.chain).c_str(), t.holds ? "tient     " : "ne tient pas", t.fps,
                 t.target_fps, t.low_fps, t.lost_percent, t.chain == done->chosen ? "  ← recommandé" : "");
  return 0;
}

// --register, --unregister : l'inscription auprès du système, puis la sortie.
int run_registration(ajp::app::play::Registration registration, const ajp::app::play::PlayerPlace& place) {
  const auto done = registration == ajp::app::play::Registration::Register
                        ? ajp::app::play::register_player(place)
                        : ajp::app::play::unregister_player(place);
  if (!done) return fail(done.error());
  std::fprintf(stdout, "%s\n", done->c_str());
  return 0;
}

// Le journal de la séance ; rend son fichier (vide : stderr seul). trial_dir : --data-dir, où vont alors
// réglages et journal (les essais du benchmark).
std::filesystem::path open_journal(const std::filesystem::path& home, const std::filesystem::path& data,
                                   const std::filesystem::path& trial_dir) {
  // Le journal va aussi dans <données>/ajp.log, recommencé à chaque lancement : ouvert depuis le
  // Finder ou l'Explorateur, le lecteur n'a pas de terminal, et c'est le fichier à joindre à un
  // rapport. Sans fichier (dossier en lecture seule, chemin hors de la page de code), stderr seul.
  // Chaque ligne y entre nettoyée (P1.32) : le dossier personnel devient « ~ ».
  ajp::set_log_home(ajp::prepare::utf8_path(home));
  std::vector<spdlog::sink_ptr> sinks{
      ajp::redacting_sink(std::make_shared<spdlog::sinks::stderr_color_sink_mt>())};
  // B36-08 : journal et minidumps dans le dossier local (pas dans le profil itinérant de Windows) ; ceux
  // d'une version d'avant y passent, ce qui est dit une fois le journal ouvert.
  std::vector<std::string> moved;
  std::filesystem::path diagnostics = trial_dir.empty() ? data : trial_dir;
  if (trial_dir.empty() && !data.empty()) {
    diagnostics = ajp::app::play::local_data(data);
    ajp::app::play::move_diagnostics(data, diagnostics,
                                     [&](const std::string& line) { moved.push_back(line); });
  }
  const std::filesystem::path log_file =
      diagnostics.empty() ? std::filesystem::path() : diagnostics / "ajp.log";
  if (!log_file.empty()) {
    // Borné pendant la séance (P2.43) : passé 32 Mo, ajp.log devient ajp.1.log (puis ajp.2.log), une soirée
    // de lecture ne remplit plus le disque.
    constexpr std::size_t kLogBytes = std::size_t{32} * 1024 * 1024;
    constexpr std::size_t kLogParts = 2;
    ajp::rotate_session_logs(log_file, kKeptLogs, kLogParts);
    // Ouvert par son chemin (UTF-16 sous Windows) : un dossier de données au nom japonais ne fait plus
    // partir le journal sans un mot (B33-06).
    if (spdlog::sink_ptr file_sink = ajp::log_file_sink(log_file, kLogBytes, kLogParts))
      sinks.push_back(ajp::redacting_sink(std::move(file_sink)));
  }
  const bool logs_to_file = sinks.size() > 1;
  ajp::install_log_sinks(std::move(sinks), spdlog::level::info, /*asynchronous=*/true);  // 1.11
  log_open() = true;
  if (logs_to_file) {
    spdlog::flush_every(std::chrono::seconds(1));  // un arrêt brutal perd au plus une seconde
    ajp::log_channel("app")->info("journal : {} · ajp {}", ajp::core::to_utf8(log_file.filename()),
                                  ajp::app::play::installed_version());
  } else if (!log_file.empty()) {
    ajp::log_channel("app")->warn("journal : {} ne s'ouvre pas, stderr seulement",
                                  ajp::core::to_utf8(log_file.filename()));
  }
  for (const std::string& line : moved) ajp::log_channel("app")->info("données : {}", line);
  return log_file;
}

// B20-06 : un rôle refusé par le système (MMCSS, rtkit) se dit une fois au journal ; sans cela, une cadence
// qui dérive sous charge ne se distinguait pas d'un défaut du lecteur.
void report_role_refusals() {
  ajp::set_role_refusal_reporter([](ajp::ThreadRole role, const char* what) noexcept {
    try {
      ajp::log_channel("app")->warn("rôle {} du thread : {}",
                                    role == ajp::ThreadRole::Background ? "de fond"
                                    : role == ajp::ThreadRole::Audio    ? "audio"
                                                                        : "de lecture",
                                    what);
    } catch (...) {  // NOLINT(bugprone-empty-catch) : le journal a échoué, le rôle reste tel quel
    }
  });
}

// Le lecteur, puis l'attente d'une écriture que la fin d'une vidéo n'a pas attendue (disque lent) : elle
// finit avant que le processus ne parte.
std::expected<void, ajp::Refusal> play_then_settle(const ajp::app::play::PlayOptions& options,
                                                   const ajp::app::play::PlayerPaths& paths) {
  auto played = ajp::app::play::play(options, paths);
  ajp::app::play::LatestWriter::wait_abandoned(std::chrono::seconds(5));
  return played;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity,readability-function-size)
int run_player(int argc, char** argv) {
  // Les trois voies CUDA, le flux de décodage, celui des copies et les flux auxiliaires de TensorRT se
  // partagent 8 files matérielles par défaut : des flux sans lien s'y alignent. 32 en donne une à chacun,
  // sous Linux comme sous Windows (où le CRT de MSVC reporte la variable dans l'environnement du processus,
  // que lit nvcuda.dll). Lue au premier appel CUDA : posée ici, avant toute bibliothèque ; une valeur de
  // l'utilisateur prime.
  SDL_setenv_unsafe("CUDA_DEVICE_MAX_CONNECTIONS", "32", 0);
#ifdef _WIN32
  // Les DLL qu'une bibliothèque chargée demande par son seul nom (TensorRT, ses ressources, NVRTC) ne se
  // cherchent plus dans le PATH ni le dossier courant (A20-07) : dossier du lecteur, System32, et les
  // dossiers que core/dynamic_library.cpp ajoute pour chaque bibliothèque ouverte.
  SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
  // Minuteur à 1 ms : les attentes courtes (vidange de l'export, démarrage) ne s'arrondissent pas
  // aux 15,6 ms par défaut de Windows. La cadence de lecture suit, elle, le swapchain.
  timeBeginPeriod(1);
#ifdef PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION
  // Windows 11 ignore sinon la résolution demandée d'un processus dont la fenêtre est masquée, et
  // peut le brider en économie d'énergie : la lecture n'est pas un travail de fond.
  PROCESS_POWER_THROTTLING_STATE throttling{.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION,
                                            .ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED |
                                                           PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION,
                                            .StateMask = 0};
  SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &throttling, sizeof(throttling));
#endif
#endif
  ajp::app::play::leave_lone_console();
  std::vector<std::string_view> args(argv + 1, argv + argc);
  if (!args.empty() && args.front() == "bench") {
    args.erase(args.begin());
    return run_bench_command(args);
  }
  if (!args.empty() && args.front() == "calibrate") {
    args.erase(args.begin());
    return run_calibrate_command(args);
  }
  if (!args.empty() && args.front() == "compare") {
    args.erase(args.begin());
    return run_compare_command(args);
  }
  if (!args.empty() && args.front() == "play") args.erase(args.begin());
  const auto options = ajp::app::play::parse_play_options(args);
  if (!options) {
    std::fputs(ajp::app::play::play_usage().c_str(), stderr);
    return fail_arguments(options.error(), args);
  }
  if (options->help) {
    std::fputs(ajp::app::play::play_usage(options->debug).c_str(), stdout);
    return 0;
  }
  quiet_failures() = !options->keys.empty() || options->snapshot || options->bench_out || options->hidden;
  const auto base = sdl_path(SDL_GetBasePath());
  char* pref = SDL_GetPrefPath("ajp", "ajp");  // avant SDL_Init comme permis : l'inscription s'en sert
  const auto data = sdl_path(pref);
  SDL_free(pref);
  if (data.empty())  // A18-07 : sans dossier de données, play.toml et engines/ iraient dans le dossier
                     // courant
    return fail_shown({ajp::RefusalCode::FileUnwritable, std::string("SDL_GetPrefPath : ") + SDL_GetError()},
                      *options);
  const ajp::app::play::PlayerPlace place{.executable = base / std::string(kExecutable),
                                          .home = sdl_path(SDL_GetUserFolder(SDL_FOLDER_HOME)),
                                          .data = ajp::app::play::user_data(data),
                                          .app_data = data};
  // La langue du système, avant tout réglage : les langues préférées par défaut la prennent.
  ajp::app::play::set_system_language(
      ajp::app::play::pick_menu_language(ajp::app::play::system_locales(place.home)));
  if (options->registration != ajp::app::play::Registration::None)
    return run_registration(options->registration, place);
  // Un seul lecteur (P2.13) : un double-clic sur une vidéo l'ouvre dans la fenêtre déjà là ; rendu avant
  // chaque relance.
  std::shared_ptr<ajp::app::play::InstanceListener> instance;
  if (!options->data_dir && !options->bench_out && !data.empty() && ajp::app::play::shares_instance(args)) {
    std::vector<std::string> files;
    std::error_code absolute;
    if (!options->file.empty())
      files.push_back(ajp::prepare::utf8_path(std::filesystem::absolute(options->file, absolute)));
    // C31-03 : un lecteur relancé prend la place de l'ancien, même s'il tient encore le canal.
    auto claim = ajp::app::play::claim_instance(data, ajp::app::play::instance_key(data), files,
                                                options->relaunched != ajp::app::play::RelaunchedFor::None);
    if (claim.forwarded) return 0;
    instance = std::move(claim.listener);
  }
  // --data-dir : réglages, profils et journal ailleurs (les essais du benchmark) ; moteurs et modèles
  // restent.
  std::filesystem::path settings_dir = data;
  if (options->data_dir) {
    settings_dir = *options->data_dir;
    std::error_code made;
    std::filesystem::create_directories(settings_dir, made);
  }
  const std::filesystem::path log_file =
      open_journal(place.home, data, options->data_dir ? settings_dir : std::filesystem::path());
  start_diagnostics(settings_dir, log_file, options->relaunched);
  report_role_refusals();
  // Le thread principal présente les images : il passe devant les travaux de fond (core/thread_role.h).
  ajp::set_thread_role(ajp::ThreadRole::Playback);
  if (options->bake_looks) {
    const auto baked = ajp::app::play::bake_looks(ajp::app::play::paths::play_settings(settings_dir),
                                                  ajp::app::play::local_data(data) / "luts",  // A18-06
                                                  *options->bake_looks);
    if (!baked) return fail(baked.error());
    return 0;
  }
  prepare_libraries();
  if (!SDL_Init(SDL_INIT_VIDEO))
    return fail_shown({ajp::RefusalCode::LibraryUnavailable, std::string("SDL_Init : ") + SDL_GetError()},
                      *options);
  // Linux : le thread principal et ceux de la lecture passent devant les travaux de fond (rtkit au besoin).
  {
    const bool raised = ajp::set_thread_priority_raiser(
        []() noexcept { return SDL_SetCurrentThreadPriority(SDL_THREAD_PRIORITY_HIGH); });
#ifdef __linux__
    ajp::log_channel("app")->info("priorité haute : {}", raised ? "accordée" : "refusée");
#else
    static_cast<void>(raised);
#endif
  }
  // L'audio est facultatif : sans serveur ni carte son (machine virtuelle, session sans PipeWire), la vidéo
  // joue sur l'horloge hôte (le sink audio retente l'initialisation et dit son refus) au lieu de ne pas
  // démarrer.
  // C18-05 : la connexion à PipeWire ou WASAPI se fait avant la fenêtre ; sa durée se dit quand elle se voit.
  const auto audio_from = std::chrono::steady_clock::now();
  if (!SDL_InitSubSystem(SDL_INIT_AUDIO))
    ajp::log_channel("app")->warn("audio indisponible, lecture sans son : {}", SDL_GetError());
  if (const auto took = std::chrono::steady_clock::now() - audio_from; took > std::chrono::milliseconds(100))
    ajp::log_channel("app")->info("audio prêt en {} ms",
                                  std::chrono::duration_cast<std::chrono::milliseconds>(took).count());
  const ajp::app::play::DataPlaces places =
      ajp::app::play::settle_data(data, place, !options->data_dir && !options->bench_out);
  const std::filesystem::path& heavy = places.heavy;
  const std::filesystem::path& cache = places.cache;
  bool engines_updated = false;  // un moteur mis à jour ou le GPU perdu : le lecteur se relance (play())
  auto relaunch_reason = ajp::app::play::RelaunchedFor::EngineUpdated;
  ajp::app::play::PlayerPaths paths{
      .home = place.home,
      .data = data,  // le dossier des données du lecteur
      .executable = place.executable,
      .engines = cache / "engines",
      .heavy = heavy,
      .cache = cache,
      .settings = settings_dir,
      .engine_index = base / "engines.lock",
      .models = base / "models",
      .relaunch = &engines_updated,
      .relaunch_reason = &relaunch_reason,
  };
  // TensorRT d'abord (cartes NVIDIA) et Core ML (Apple), qui refusent les autres fabricants, puis Vulkan (le
  // recours partout) ; chacun à côté de l'exécutable, puis dans l'arbre de construction.
  const auto beside = [&](std::string_view backend) {
    for (const std::filesystem::path& dir : module_dirs(base))
      paths.backends.push_back(
          dir / (std::string("ajp_backend_") + std::string(backend) + std::string(kModuleSuffix)));
  };
  beside("tensorrt");
#ifdef AJP_BACKEND_TENSORRT
  paths.backends.emplace_back(AJP_BACKEND_TENSORRT);
#endif
  beside("coreml");  // Mac : Core ML, qui refuse les autres appareils
#ifdef AJP_BACKEND_COREML
  paths.backends.emplace_back(AJP_BACKEND_COREML);
#endif
  beside("vulkan");
#ifdef AJP_BACKEND_VULKAN
  paths.backends.emplace_back(AJP_BACKEND_VULKAN);
#endif
  before_play(paths.engines, place.executable, argc, argv, log_file);
  const auto played = play_then_settle(*options, paths);
  instance.reset();  // avant SDL_Quit : le fil du canal pousse dans la file de SDL
  SDL_Quit();
  if (played && engines_updated) return relaunch(place.executable, argc, argv, relaunch_reason, log_file);
  if (played) return 0;
  // C18-09 : la fenêtre fermée pendant une attente (moteur, sonde) est une sortie voulue : ni erreur ni
  // code 1.
  if (played.error().code == ajp::RefusalCode::Cancelled) {
    ajp::log_channel("app")->info("fermé pendant l'attente : {}", played.error().detail);
    return 0;
  }
  // Par le journal, qui l'écrit aussi sur stderr : le refus reste dans ajp.log.
  ajp::log_channel("app")->error("{} {}", ajp::name(played.error().code), played.error().detail);
  // Lancé par un double-clic, personne ne lit stderr : la fenêtre ne se ferme pas sans un mot. Pas
  // de boîte pendant un essai scripté (--keys, --snapshot, banc), qui doit finir seul, ni pour une
  // annulation voulue.
  if (!quiet_failures()) {
    namespace i18n = ajp::app::play::i18n;
    std::string text =
        i18n::tr_format("Playback stopped: {}.", ajp::app::play::visible_error(played.error()));
    if (!log_file.empty())
      text += "\n\n" + i18n::tr_format("Details in {}", ajp::prepare::utf8_path(log_file));
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "ajp", text.c_str(), nullptr);
  }
  return 1;
}

}  // namespace

// Une exception qui sort du lecteur, ou un std::terminate, laisse sa cause dans ajp.log et vide la file
// asynchrone du journal avant de s'arrêter (P2.1).
int main(int argc, char** argv) {
  ajp::install_terminate_flush();
  ajp::install_system_thread_names();  // Nsight, gdb et les vidages nomment chaque thread (A20-11)
  bool threw = true;                   // remis à faux par run_player qui rend
  const int code = ajp::run_logged([&] {
    const int returned = run_player(argc, argv);
    threw = false;
    return returned;
  });
  if (threw) {
    // B31-10 : une exception (mémoire pleine) ne ferme plus la fenêtre sans un mot, et la séance reste
    // marquée ouverte : le lancement suivant joint ce journal au rapport.
    if (!quiet_failures()) {
      const std::string text = ajp::app::play::i18n::tr("The player stopped on an internal error.");
      SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "ajp", text.c_str(), nullptr);
    }
  } else {
    ajp::app::play::end_session();  // une sortie qui revient jusqu'ici n'est pas un plantage
  }
  // Une sonde de fichier bloquée sur un partage muet (run_pumped_for) : un instant pour finir, sinon le
  // processus se termine sans elle, avant qu'elle ne journalise après close_log (A30-05).
  if (!ajp::app::play::settle_abandoned_threads(std::chrono::seconds(2))) {
    ajp::log_channel("app")->warn("sonde d'un fichier toujours bloquée : sortie sans l'attendre");
    ajp::close_log();
    std::quick_exit(code);
  }
  ajp::close_log();
  return code;
}
