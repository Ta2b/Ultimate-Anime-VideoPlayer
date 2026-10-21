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
#include <cstdio>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "app/play/default_player.h"
#include "app/play/options.h"
#include "app/play/player.h"
#include "core/log.h"
#include "prepare/utf8.h"

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

}  // namespace

int main(int argc, char** argv) {
  ajp::app::play::leave_lone_console();
  std::vector<std::string_view> args(argv + 1, argv + argc);
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
  // Le journal va aussi dans <données>/ajp.log, recommencé à chaque lancement : ouvert depuis le
  // Finder ou l'Explorateur, le lecteur n'a pas de terminal, et c'est le fichier à joindre à un
  // rapport. Sans fichier (dossier en lecture seule, chemin hors de la page de code), stderr seul.
  std::vector<spdlog::sink_ptr> sinks{std::make_shared<spdlog::sinks::stderr_color_sink_mt>()};
  const std::filesystem::path log_file = data.empty() ? std::filesystem::path() : data / "ajp.log";
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
