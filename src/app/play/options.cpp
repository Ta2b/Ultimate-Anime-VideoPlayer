#include "app/play/options.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <format>
#include <system_error>
#include <utility>

namespace ajp::app::play {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

template <class T>
std::optional<T> number(std::string_view text) {
  T value{};
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size()) return std::nullopt;
  return value;
}

// Secondes, décimales permises (« 12.5 »), en microsecondes dans limits::kSeconds.
std::expected<std::int64_t, Refusal> seconds(std::string_view option, std::string_view text) {
  const auto value = number<double>(text);
  if (!value || !std::isfinite(*value) || *value < 0.0 ||
      !limits::kSeconds.contains(static_cast<std::int64_t>(*value))) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("{} : « {} » n'est pas une durée en secondes "
                              "entre {} et {}",
                              option, text, limits::kSeconds.lo, limits::kSeconds.hi));
  }
  return std::llround(*value * 1e6);
}

std::expected<void, Refusal> window(std::string_view text, PlayOptions& options) {
  const auto x = text.find('x');
  const auto width = x == std::string_view::npos ? std::nullopt : number<int>(text.substr(0, x));
  const auto height = x == std::string_view::npos ? std::nullopt : number<int>(text.substr(x + 1));
  if (!width || !height || !limits::kWindowSide.contains(*width) || !limits::kWindowSide.contains(*height)) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("--window : « {} » n'est pas une taille LxH, côtés de {} à {}", text,
                              limits::kWindowSide.lo, limits::kWindowSide.hi));
  }
  options.window_width = *width;
  options.window_height = *height;
  return {};
}

// « 3:Right,5.5:Space » : instants en secondes et noms de touches, dans l'ordre.
std::expected<void, Refusal> keys(std::string_view text, PlayOptions& options) {
  while (!text.empty()) {
    const auto comma = text.find(',');
    const std::string_view item = text.substr(0, comma);
    text = comma == std::string_view::npos ? std::string_view{} : text.substr(comma + 1);
    const auto colon = item.find(':');
    if (colon == std::string_view::npos || colon + 1 == item.size()) {
      return refuse(RefusalCode::FieldOutOfRange, std::format("--keys : « {} » n'est pas S:touche", item));
    }
    auto at = seconds("--keys", item.substr(0, colon));
    if (!at) return std::unexpected(std::move(at.error()));
    options.keys.emplace_back(*at, std::string(item.substr(colon + 1)));
  }
  return {};
}

// Une option à valeur : « --nom=valeur » ou « --nom valeur ». Rend la valeur, ou rien si
// args[i] n'est pas cette option ; avance i quand la valeur est l'argument suivant.
std::expected<std::optional<std::string_view>, Refusal> valued(std::span<const std::string_view> args,
                                                               std::size_t& i, std::string_view name) {
  const std::string_view arg = args[i];
  if (!arg.starts_with(name)) return std::nullopt;
  const std::string_view rest = arg.substr(name.size());
  if (rest.starts_with('=')) return rest.substr(1);
  if (!rest.empty()) return std::nullopt;  // « --modelX » n'est pas « --model »
  if (i + 1 >= args.size())
    return refuse(RefusalCode::FieldMissing, std::format("{} : valeur absente", name));
  return args[++i];
}

}  // namespace

std::expected<PlayOptions, Refusal> parse_play_options(std::span<const std::string_view> args) {
  PlayOptions options;
  bool only_files = false;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string_view arg = args[i];
    if (only_files || !arg.starts_with("--")) {
      if (!options.file.empty()) {
        return refuse(RefusalCode::OptionUnknown,
                      std::format("un seul fichier à la fois : « {} » en trop", arg));
      }
      options.file = std::filesystem::path(arg);
      continue;
    }
    if (arg == "--") {
      only_files = true;
    } else if (arg == "--help") {
      options.help = true;
    } else if (arg == "--no-upscale") {
      options.upscale = false;
    } else if (arg == "--fullscreen") {
      options.fullscreen = true;
    } else {
      std::expected<void, Refusal> applied;
      bool known = false;
      for (const std::string_view name :
           {"--model", "--window", "--start", "--seconds", "--fullscreen-after", "--stats", "--snapshot-at",
            "--snapshot", "--keys", "--work-height", "--rife-model", "--rife"}) {
        auto value = valued(args, i, name);
        if (!value) return std::unexpected(std::move(value.error()));
        if (!*value) continue;
        known = true;
        const std::string_view text = **value;
        if (name == "--model") {
          options.model = std::filesystem::path(text);
        } else if (name == "--window") {
          applied = window(text, options);
        } else if (name == "--stats") {
          options.stats = std::filesystem::path(text);
        } else if (name == "--keys") {
          applied = keys(text, options);
        } else if (name == "--snapshot") {
          options.snapshot = std::filesystem::path(text);
        } else if (name == "--rife-model") {
          options.rife_model = std::filesystem::path(text);
        } else if (name == "--rife") {
          const auto factor = number<int>(text);
          if (!factor || !limits::kRifeFactor.contains(*factor)) {
            return refuse(RefusalCode::FieldOutOfRange,
                          std::format("--rife : « {} » n'est pas un facteur entre {} et {}", text,
                                      limits::kRifeFactor.lo, limits::kRifeFactor.hi));
          }
          options.rife_factor = *factor;
        } else if (name == "--work-height") {
          const auto lines = number<int>(text);
          if (!lines || !limits::kWorkHeight.contains(*lines)) {
            return refuse(RefusalCode::FieldOutOfRange,
                          std::format("--work-height : « {} » n'est pas un nombre de lignes entre {} et {}",
                                      text, limits::kWorkHeight.lo, limits::kWorkHeight.hi));
          }
          options.work_height = *lines;
        } else {
          auto us = seconds(name, text);
          if (!us) return std::unexpected(std::move(us.error()));
          if (name == "--start") options.start_us = *us;
          if (name == "--seconds") options.stop_after_us = *us;
          if (name == "--fullscreen-after") options.fullscreen_after_us = *us;
          if (name == "--snapshot-at") options.snapshot_at_us = *us;
        }
        break;
      }
      if (!applied) return std::unexpected(std::move(applied.error()));
      if (!known) return refuse(RefusalCode::OptionUnknown, std::format("option inconnue : {}", arg));
    }
  }
  if (options.file.empty() && !options.help) {
    return refuse(RefusalCode::FieldMissing, "ajp play : fichier vidéo absent");
  }
  return options;
}

WorkSize work_size(int width, int height, int work_height) {
  if (work_height <= 0 || height <= work_height || width <= 0 || height <= 0) return {width, height};
  const auto scaled = static_cast<int>(std::lround(static_cast<double>(width) * work_height / height));
  return {std::max(2, scaled + (scaled % 2)), work_height};
}

std::string play_usage() {
  return std::format(
      "ajp play <fichier> [options]\n"
      "  --model=FICHIER.onnx   modèle SPAN ×2 (défaut : {})\n"
      "  --no-upscale           image source telle quelle\n"
      "  --window=LxH           taille de la fenêtre (défaut 1280x720)\n"
      "  --fullscreen           plein écran dès l'ouverture\n"
      "  --start=S              position de départ, en secondes\n"
      "  --seconds=S            quitte après S secondes de lecture\n"
      "  --fullscreen-after=S   passe en plein écran après S secondes\n"
      "  --stats=FICHIER        temps de chaque image présentée\n"
      "  --snapshot=FICHIER     la fenêtre en PPM, après --snapshot-at=S secondes (3)\n"
      "  --keys=S:touche,…      touches pressées d'office (essais), ex. 3:Right,5:Space\n"
      "  --work-height=N        réduit la source à N lignes avant SPAN (défaut 720 ; 0 jamais)\n"
      "  --rife=N               interpolation RIFE ×N avant SPAN (défaut {} ; 0 aucune)\n"
      "  --rife-model=FICHIER   modèle RIFE (défaut : {})\n"
      "Touches : espace pause, flèches seek 5 s, f plein écran, q quitter.\n",
      kDefaultModelFile, kDefaultRifeFactor, kDefaultRifeFile);
}

std::vector<std::filesystem::path> default_model_candidates(const std::filesystem::path& home) {
  return {
      home / ".local/share/AnimeJaNai-HDR-linux/animejanai/onnx" / kDefaultModelFile,
      home /
          "Documents/mpv-AnimeJaNai-HDR/BuildMpvUpscale2xAnimeJaNai/mpv-upscale-2x_animejanai/animejanai/"
          "onnx" /
          kDefaultModelFile,
  };
}

std::vector<std::filesystem::path> default_rife_candidates(const std::filesystem::path& home) {
  return {home / ".local/share/AnimeJaNai-HDR-linux/animejanai/rife" / kDefaultRifeFile,
          home / ".local/share/AnimeJaNai/animejanai/rife" / kDefaultRifeFile};
}

namespace {

// Le fichier donné par l'option (il doit exister), sinon le premier candidat présent.
std::expected<std::filesystem::path, Refusal> find_file(const std::optional<std::filesystem::path>& given,
                                                        std::string_view option, std::string_view what,
                                                        std::span<const std::filesystem::path> candidates) {
  std::error_code error;
  if (given) {
    if (std::filesystem::is_regular_file(*given, error)) return *given;
    return refuse(RefusalCode::ModelUnreadable, std::format("{} : {} introuvable", option, given->string()));
  }
  std::string tried;
  for (const auto& path : candidates) {
    if (std::filesystem::is_regular_file(path, error)) return path;
    tried += std::format("\n  {}", path.string());
  }
  return refuse(RefusalCode::ModelUnreadable, std::format("{} par défaut introuvable :{}", what, tried));
}

}  // namespace

std::expected<std::filesystem::path, Refusal> find_model(const PlayOptions& options,
                                                         std::span<const std::filesystem::path> candidates) {
  return find_file(options.model, "--model", "modèle AnimeJaNai", candidates);
}

std::expected<std::filesystem::path, Refusal> find_rife_model(
    const PlayOptions& options, std::span<const std::filesystem::path> candidates) {
  return find_file(options.rife_model, "--rife-model", "modèle RIFE", candidates);
}

}  // namespace ajp::app::play
