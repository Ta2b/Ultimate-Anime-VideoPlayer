// ajp : « ajp play <fichier> [options] », ou « ajp <fichier> ».
//
// Tranche verticale : la ligne de commande complète (export, bench, engines…) vient du
// chantier app-shell ; ici, seulement la lecture.
#include <SDL3/SDL_filesystem.h>
#include <SDL3/SDL_init.h>
#include <spdlog/sinks/stdout_color_sinks.h>

extern "C" {
#include <libavutil/log.h>
}

#include <cstdio>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "app/play/options.h"
#include "app/play/player.h"
#include "core/log.h"

namespace {

std::filesystem::path sdl_path(const char* path) {
  return path != nullptr ? std::filesystem::path(path) : std::filesystem::path();
}

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
  ajp::install_log_sinks({std::make_shared<spdlog::sinks::stderr_color_sink_mt>()}, spdlog::level::info);
  av_log_set_level(AV_LOG_ERROR);  // FFmpeg écrit sur stderr ; ses avertissements de sonde sont du bruit
  if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO)) {
    return fail({ajp::RefusalCode::LibraryUnavailable, std::string("SDL_Init : ") + SDL_GetError()});
  }
  char* pref = SDL_GetPrefPath("ajp", "ajp");
  const auto data = sdl_path(pref);
  SDL_free(pref);
  const auto base = sdl_path(SDL_GetBasePath());
  ajp::app::play::PlayerPaths paths{
      .home = sdl_path(SDL_GetUserFolder(SDL_FOLDER_HOME)),
      .engines = data / "engines",
      .engine_index = base / "engines.lock",
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
  return played ? 0 : fail(played.error());
}
