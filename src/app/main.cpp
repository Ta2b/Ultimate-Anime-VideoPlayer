// ajp : « ajp play <fichier> [options] », ou « ajp <fichier> » ; « ajp --register » l'inscrit
// comme lecteur vidéo auprès du système (default_player.h).
//
// Tranche verticale : la ligne de commande complète (export, bench, engines…) vient du
// chantier app-shell ; ici, seulement la lecture.
#include <SDL3/SDL_filesystem.h>
#include <SDL3/SDL_init.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

extern "C" {
#include <libavutil/log.h>
}

#include <chrono>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif
#include <cstdio>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

#include "app/play/auto_bench_run.h"
#include "app/play/default_player.h"
#include "app/play/options.h"
#include "app/play/player.h"
#include "app/play/settings.h"
#include "app/play/updater.h"
#include "core/log.h"
#include "core/update_manifest.h"
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

int fail(const ajp::Refusal& refusal) {
  std::fprintf(stderr, "ajp : %s %s\n", std::string(ajp::name(refusal.code)).c_str(), refusal.detail.c_str());
  return 1;
}

// Relance le lecteur avec les mêmes arguments, après une mise à jour : il remplace ce processus (POSIX) ou
// démarre à sa place (Windows, où un exécutable en service ne s'écrase pas : l'ancien est renommé .old).
int relaunch(const std::filesystem::path& executable, int argc, char** argv) {
#ifdef _WIN32
  std::vector<std::string> arguments;
  for (int i = 1; i < argc; ++i) arguments.emplace_back(argv[i]);
  std::vector<const char*> pointers{executable.string().c_str()};
  for (const std::string& argument : arguments) pointers.push_back(argument.c_str());
  pointers.push_back(nullptr);
  return _spawnv(_P_NOWAIT, executable.string().c_str(), pointers.data()) == -1 ? 1 : 0;
#else
  std::vector<char*> arguments{const_cast<char*>(executable.c_str())};
  for (int i = 1; i < argc; ++i) arguments.push_back(argv[i]);
  arguments.push_back(nullptr);
  execv(executable.c_str(), arguments.data());
  return 1;  // execv ne revient que sur un échec
#endif
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
  const play::BenchPlan plan{.executable = sdl_path(SDL_GetBasePath()) / std::string(kExecutable),
                             .video = options->file,
                             .work_dir = work,
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
  if (options->report) {
    std::ofstream out(*options->report, std::ios::binary);
    out << report;
  }
  return 0;
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
                             .seconds = options->seconds,
                             .fullscreen = options->fullscreen};
  const auto report = play::evaluate_machine(plan, data, std::stop_token{}, [](const std::string& step) {
    std::fprintf(stdout, "%s\n", step.c_str());
    std::fflush(stdout);
  });
  std::filesystem::remove_all(work, error);
  if (!report) return fail(report.error());
  std::fputs(report->c_str(), stdout);
  if (options->report) {
    std::ofstream out(*options->report, std::ios::binary);
    out << *report;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
#if defined(__linux__)
  // Les trois voies CUDA, le flux de décodage, celui des copies et le pont de Maxine se partagent
  // 8 files matérielles par défaut : des flux sans lien s'y alignent. 32 en donne une à chacun.
  // Lue au premier appel CUDA : posée ici, avant toute bibliothèque ; une valeur de l'utilisateur
  // prime.
  SDL_setenv_unsafe("CUDA_DEVICE_MAX_CONNECTIONS", "32", 0);
#endif
#ifdef _WIN32
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
  if (!args.empty() && args.front() == "compare") {
    args.erase(args.begin());
    return run_compare_command(args);
  }
  if (!args.empty() && args.front() == "play") args.erase(args.begin());
  const auto options = ajp::app::play::parse_play_options(args);
  if (!options) {
    std::fputs(ajp::app::play::play_usage().c_str(), stderr);
    return fail(options.error());
  }
  if (options->help) {
    std::fputs(ajp::app::play::play_usage().c_str(), stdout);
    return 0;
  }
  const auto base = sdl_path(SDL_GetBasePath());
  char* pref = SDL_GetPrefPath("ajp", "ajp");  // avant SDL_Init comme permis : l'inscription s'en sert
  const auto data = sdl_path(pref);
  SDL_free(pref);
  const ajp::app::play::PlayerPlace place{.executable = base / std::string(kExecutable),
                                          .home = sdl_path(SDL_GetUserFolder(SDL_FOLDER_HOME)),
                                          .data = ajp::app::play::user_data(data)};
  if (options->registration != ajp::app::play::Registration::None) {
    const auto done = options->registration == ajp::app::play::Registration::Register
                          ? ajp::app::play::register_player(place)
                          : ajp::app::play::unregister_player(place);
    if (!done) return fail(done.error());
    std::fprintf(stdout, "%s\n", done->c_str());
    return 0;
  }
  // Une mise à jour préparée (mode Automatique) s'applique ici, avant tout, puis le lecteur se relance.
  if (!options->data_dir && !options->bench_out) {
    const ajp::app::play::UpdatePaths update{.data = data, .install = base, .executable = place.executable};
    ajp::app::play::UpdateState state = ajp::app::play::load_update_state(data / "update.toml");
    const auto saved = ajp::app::play::load_settings(data / "play.toml");
    if (!state.staged_version.empty() && saved && saved->update_mode == "auto" &&
        ajp::core::compare_versions(state.staged_version, ajp::app::play::installed_version()).value_or(0) >
            0 &&
        ajp::app::play::update_managed_reason(update).empty()) {
      const auto applied = ajp::app::play::apply_staged_update(update, state);
      state.last_error = applied ? std::string() : applied.error().detail;
      if (!applied)
        state.staged_version
            .clear();  // une archive qui ne s'applique pas ne se réessaie pas à chaque lancement
      if (const auto kept = ajp::app::play::save_update_state(data / "update.toml", state); !kept)
        std::fprintf(stderr, "ajp : update.toml : %s\n", kept.error().detail.c_str());
      if (applied) {
        std::fprintf(stderr, "ajp : %s\n", applied->c_str());
        return relaunch(place.executable, argc, argv);
      }
      std::fprintf(stderr, "ajp : mise à jour impossible : %s\n", applied.error().detail.c_str());
    }
  }
  // Le journal va aussi dans <données>/ajp.log, recommencé à chaque lancement : ouvert depuis le
  // Finder ou l'Explorateur, le lecteur n'a pas de terminal, et c'est le fichier à joindre à un
  // rapport. Sans fichier (dossier en lecture seule, chemin hors de la page de code), stderr seul.
  std::vector<spdlog::sink_ptr> sinks{std::make_shared<spdlog::sinks::stderr_color_sink_mt>()};
  // --data-dir : réglages, profils et journal ailleurs (les essais du benchmark) ; moteurs et modèles
  // restent.
  std::filesystem::path settings_dir = data;
  if (options->data_dir) {
    settings_dir = *options->data_dir;
    std::error_code made;
    std::filesystem::create_directories(settings_dir, made);
  }
  const std::filesystem::path log_file =
      settings_dir.empty() ? std::filesystem::path() : settings_dir / "ajp.log";
  if (!log_file.empty()) {
    try {
      sinks.push_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file.string(), true));
    } catch (const std::exception&) {
      sinks.resize(1);
    }
  }
  const bool logs_to_file = sinks.size() > 1;
  ajp::install_log_sinks(std::move(sinks), spdlog::level::info);
  if (logs_to_file) {
    spdlog::flush_every(std::chrono::seconds(1));  // un arrêt brutal perd au plus une seconde
    ajp::log_channel("app")->info("journal : {}", log_file.string());
  }
  av_log_set_level(AV_LOG_ERROR);  // FFmpeg écrit sur stderr ; ses avertissements de sonde sont du bruit
  if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO)) {
    return fail({ajp::RefusalCode::LibraryUnavailable, std::string("SDL_Init : ") + SDL_GetError()});
  }
  ajp::app::play::PlayerPaths paths{
      .home = place.home,
      .data = place.data,
      .executable = place.executable,
      .engines = data / "engines",
      .settings = settings_dir,
      .engine_index = base / "engines.lock",
      .models = base / "models",
  };
  // TensorRT d'abord (cartes NVIDIA), MIGraphX (AMD), OpenVINO (Intel) et Core ML (Apple), qui
  // refusent les autres fabricants, puis Vulkan (le recours partout) ; chacun à côté de l'exécutable, puis
  // dans l'arbre de construction.
  const auto beside = [&](std::string_view backend) {
    for (const std::filesystem::path& dir : module_dirs(base))
      paths.backends.push_back(
          dir / (std::string("ajp_backend_") + std::string(backend) + std::string(kModuleSuffix)));
  };
  beside("tensorrt");
#ifdef AJP_BACKEND_TENSORRT
  paths.backends.emplace_back(AJP_BACKEND_TENSORRT);
#endif
  beside("migraphx");
#ifdef AJP_BACKEND_MIGRAPHX
  paths.backends.emplace_back(AJP_BACKEND_MIGRAPHX);
#endif
  beside("openvino");
#ifdef AJP_BACKEND_OPENVINO
  paths.backends.emplace_back(AJP_BACKEND_OPENVINO);
#endif
  beside("coreml");  // Mac : Core ML, qui refuse les autres appareils
#ifdef AJP_BACKEND_COREML
  paths.backends.emplace_back(AJP_BACKEND_COREML);
#endif
  beside("vulkan");
#ifdef AJP_BACKEND_VULKAN
  paths.backends.emplace_back(AJP_BACKEND_VULKAN);
#endif
  std::error_code error;
  std::filesystem::create_directories(paths.engines, error);
  const auto played = ajp::app::play::play(*options, paths);
  SDL_Quit();
  if (played) return 0;
  // Par le journal, qui l'écrit aussi sur stderr : le refus reste dans ajp.log.
  ajp::log_channel("app")->error("{} {}", ajp::name(played.error().code), played.error().detail);
  return 1;
}
