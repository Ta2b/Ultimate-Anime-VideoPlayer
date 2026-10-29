#include "app/play/options.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <format>
#include <system_error>
#include <utility>

#include "app/play/settings.h"
#include "core/charconv.h"

namespace ajp::app::play {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

template <class T>
std::optional<T> number(std::string_view text) {
  T value{};
  const auto [end, error] = core::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size()) return std::nullopt;
  return value;
}

// Secondes, décimales permises (« 12.5 »), en microsecondes dans limits::kSeconds.
std::expected<std::int64_t, Refusal> seconds(std::string_view option, std::string_view text) {
  const auto value = number<double>(text);
  // Comparé en double avant toute conversion : un double hors de l'int64 serait indéfini.
  if (!value || !std::isfinite(*value) || *value < 0.0 || *value > static_cast<double>(limits::kSeconds.hi) ||
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
    const std::string name(item.substr(colon + 1));
    // « click=XxY » : la souris arrive 0,1 s avant, comme une vraie main (ImGui n'active un
    // élément que s'il était déjà sous la souris).
    if (name.starts_with("click="))
      options.keys.emplace_back(std::max<std::int64_t>(*at - 100'000, 0), "mouse=" + name.substr(6));
    options.keys.emplace_back(*at, name);
  }
  std::ranges::stable_sort(options.keys, {}, &std::pair<std::int64_t, std::string>::first);
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
      options.given.emplace_back(arg);
    } else if (arg == "--fullscreen") {
      options.fullscreen = true;
    } else if (arg == "--hidden") {
      options.hidden = true;
    } else if (arg == "--truehdr") {
      options.true_hdr = true;
      options.given.emplace_back(arg);
    } else if (arg == "--mute") {
      options.mute = true;
      options.given.emplace_back(arg);
    } else if (arg == "--first-run") {
      options.first_run = true;
    } else if (arg == "--rife-image") {
      options.rife_image = true;
    } else if (arg == "--software-gpu") {
      options.software_gpu = true;
    } else if (arg == "--sdr") {
      options.sdr = true;
    } else if (arg == "--no-vrr") {
      options.no_vrr = true;
    } else if (arg == "--one-lane") {
      options.one_lane = true;
    } else if (arg == "--no-resume") {
      options.resume = false;
    } else if (arg == "--register" || arg == "--unregister") {
      const Registration wanted = arg == "--register" ? Registration::Register : Registration::Unregister;
      if (options.registration != Registration::None && options.registration != wanted)
        return refuse(RefusalCode::OptionUnknown, "--register et --unregister ensemble");
      options.registration = wanted;
    } else {
      std::expected<void, Refusal> applied;
      bool known = false;
      for (const std::string_view name : {"--model",
                                          "--window",
                                          "--start",
                                          "--seconds",
                                          "--fullscreen-after",
                                          "--stats",
                                          "--snapshot-at",
                                          "--snapshot-frames",
                                          "--snapshot",
                                          "--keys",
                                          "--work-height",
                                          "--rife-model",
                                          "--rife",
                                          "--truehdr-sdk",
                                          "--truehdr-peak",
                                          "--truehdr-contrast",
                                          "--truehdr-saturation",
                                          "--truehdr-gray",
                                          "--data-dir",
                                          "--bench-level",
                                          "--bench-out"}) {
        auto value = valued(args, i, name);
        if (!value) return std::unexpected(std::move(value.error()));
        if (!*value) continue;
        known = true;
        options.given.emplace_back(name);
        const std::string_view text = **value;
        if (name == "--model") {
          options.model = std::filesystem::path(text);
        } else if (name == "--window") {
          applied = window(text, options);
        } else if (name == "--data-dir") {
          options.data_dir = std::filesystem::path(text);
        } else if (name == "--bench-out") {
          options.bench_out = std::filesystem::path(text);
        } else if (name == "--bench-level") {
          const auto level = text == "auto" ? std::optional<int>(-1) : number<int>(text);
          if (!level || *level < -1 || *level >= 5)
            return refuse(RefusalCode::FieldOutOfRange,
                          std::format("--bench-level : « {} » n'est ni auto ni 0 à 4", text));
          options.bench_level = level;
        } else if (name == "--stats") {
          options.stats = std::filesystem::path(text);
        } else if (name == "--keys") {
          applied = keys(text, options);
        } else if (name == "--snapshot-frames") {
          const auto frames = number<int>(text);
          if (!frames || *frames < 1 || *frames > kMaxSnapshotFrames)
            return refuse(RefusalCode::FieldOutOfRange,
                          std::format("--snapshot-frames : « {} » hors de 1 à {}", text, kMaxSnapshotFrames));
          options.snapshot_frames = *frames;
        } else if (name == "--snapshot") {
          options.snapshot = std::filesystem::path(text);
        } else if (name == "--truehdr-sdk") {
          options.true_hdr_sdk = std::filesystem::path(text);
        } else if (name.starts_with("--truehdr-")) {
          const auto setting = number<int>(text);
          // Les bornes de play.toml (limits::), le pic à 0 pour « celui de l'écran ».
          const bool peak = name == "--truehdr-peak";
          const core::Range<int> range = peak                           ? limits::kTrueHdrPeak
                                         : name == "--truehdr-gray"     ? limits::kTrueHdrGray
                                         : name == "--truehdr-contrast" ? limits::kTrueHdrContrast
                                                                        : limits::kTrueHdrSaturation;
          if (!setting || !(range.contains(*setting) || (peak && *setting == 0))) {
            return refuse(RefusalCode::FieldOutOfRange,
                          std::format("{} : « {} » hors de {} à {}", name, text, range.lo, range.hi));
          }
          if (peak) options.true_hdr_peak = *setting;
          if (name == "--truehdr-contrast") options.true_hdr_contrast = *setting;
          if (name == "--truehdr-saturation") options.true_hdr_saturation = *setting;
          if (name == "--truehdr-gray") options.true_hdr_gray = *setting;
        } else if (name == "--rife-model") {
          options.rife_model = std::filesystem::path(text);
        } else if (name == "--rife") {
          const auto factor = parse_factor(text);
          if (!factor) {
            return refuse(
                RefusalCode::FieldOutOfRange,
                std::format("--rife : « {} » n'est pas un facteur entre {} et {} (entier ou moitié)", text,
                            limits::kRifeFactor.lo, limits::kRifeFactor.hi));
          }
          options.rife_factor = factor->num;
          options.rife_factor_den = factor->den;
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
  if (options.registration != Registration::None && !options.file.empty()) {
    return refuse(RefusalCode::OptionUnknown, "--register et --unregister s'emploient sans fichier vidéo");
  }
  return options;  // sans fichier : l'écran d'accueil (home_screen.h)
}

bool PlayOptions::was_given(std::string_view name) const {
  return std::ranges::find(given, name) != given.end();
}

WorkSize work_size(int width, int height, int work_height) {
  if (work_height <= 0 || height <= work_height || width <= 0 || height <= 0) return {width, height};
  const auto scaled = static_cast<int>(std::lround(static_cast<double>(width) * work_height / height));
  return {std::max(2, scaled + (scaled % 2)), work_height};
}

std::expected<BenchOptions, Refusal> parse_bench_options(std::span<const std::string_view> args) {
  BenchOptions options;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string_view arg = args[i];
    if (!arg.starts_with("--")) {
      if (!options.file.empty())
        return refuse(RefusalCode::OptionUnknown,
                      std::format("un seul fichier à la fois : « {} » en trop", arg));
      options.file = std::filesystem::path(arg);
    } else if (arg == "--help") {
      options.help = true;
    } else if (arg == "--windowed") {
      options.fullscreen = false;
    } else if (arg == "--visible") {
      options.silent = false;
    } else if (auto report = valued(args, i, "--report"); !report) {
      return std::unexpected(std::move(report.error()));
    } else if (*report) {
      options.report = std::filesystem::path(**report);
    } else if (auto length = valued(args, i, "--seconds"); !length) {
      return std::unexpected(std::move(length.error()));
    } else if (*length) {
      const auto value = number<int>(**length);
      if (!value || *value < 5 || *value > 120)
        return refuse(RefusalCode::FieldOutOfRange,
                      std::format("--seconds : « {} » n'est pas une durée de 5 à 120 s", **length));
      options.seconds = *value;
    } else {
      return refuse(RefusalCode::OptionUnknown, std::format("option inconnue : {}", arg));
    }
  }
  return options;  // sans fichier : la séquence de test du lecteur (media/bench_clip.h)
}

std::expected<CompareOptions, Refusal> parse_compare_options(std::span<const std::string_view> args) {
  CompareOptions options;
  const auto words = [](std::string_view text) {
    std::vector<std::string> out;
    while (!text.empty()) {
      const auto space = text.find(' ');
      if (space != 0) out.emplace_back(text.substr(0, space));
      text = space == std::string_view::npos ? std::string_view{} : text.substr(space + 1);
    }
    return out;
  };
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string_view arg = args[i];
    if (!arg.starts_with("--")) {
      if (!options.file.empty())
        return refuse(RefusalCode::OptionUnknown,
                      std::format("un seul fichier à la fois : « {} » en trop", arg));
      options.file = std::filesystem::path(arg);
      continue;
    }
    if (arg == "--help") {
      options.help = true;
      continue;
    }
    if (arg == "--windowed") {
      options.fullscreen = false;
      continue;
    }
    bool known = false;
    for (const std::string_view name :
         {"--a-settings", "--b-settings", "--a", "--b", "--report", "--seconds"}) {
      auto value = valued(args, i, name);
      if (!value) return std::unexpected(std::move(value.error()));
      if (!*value) continue;
      known = true;
      const std::string_view text = **value;
      if (name == "--a") options.a = words(text);
      if (name == "--b") options.b = words(text);
      if (name == "--a-settings") options.a_settings = std::filesystem::path(text);
      if (name == "--b-settings") options.b_settings = std::filesystem::path(text);
      if (name == "--report") options.report = std::filesystem::path(text);
      if (name == "--seconds") {
        const auto value_s = number<int>(text);
        if (!value_s || *value_s < 5 || *value_s > 120)
          return refuse(RefusalCode::FieldOutOfRange,
                        std::format("--seconds : « {} » n'est pas une durée de 5 à 120 s", text));
        options.seconds = *value_s;
      }
      break;
    }
    if (!known) return refuse(RefusalCode::OptionUnknown, std::format("option inconnue : {}", arg));
  }
  if (!options.help && options.file.empty())
    return refuse(RefusalCode::FieldMissing, "ajp compare : la vidéo à jouer manque");
  return options;
}

std::string compare_usage() {
  return "ajp compare <fichier> --a=\"OPTIONS\" --b=\"OPTIONS\" [options]   deux chaînes côte à côte, son "
         "coupé\n"
         "  --a, --b            options de « ajp play » de chaque variante, ex. --a=\"--rife=3\" "
         "--b=\"--rife=4\"\n"
         "  --a-settings=FICHIER, --b-settings=FICHIER   play.toml de départ (poses DRBA, moteur "
         "d'interpolation…)\n"
         "  --seconds=S         durée d'un essai (5 à 120, défaut 12) ; A, B, B, A : le pire des deux "
         "passages\n"
         "  --report=FICHIER    le rapport, en plus de la sortie standard ; --windowed : dans une fenêtre\n";
}

std::string bench_usage() {
  return "ajp bench [fichier] [options]   évalue la machine à chaque niveau, son coupé, fenêtre cachée ; "
         "sans fichier, "
         "la séquence de test du lecteur\n"
         "  --seconds=S      durée d'un essai (5 à 120, défaut 12)\n"
         "  --report=FICHIER le rapport, en plus de la sortie standard\n"
         "  --windowed       joue dans une fenêtre au lieu du plein écran\n"
         "  --visible        montre la fenêtre des essais (cachée par défaut)\n"
         "Le résultat est gardé dans bench.toml (carte, pilote, niveau, date) et remplace l'estimation de la "
         "carte.\n";
}

std::string play_usage() {
  return std::format(
      "ajp play [fichier] [options]   (sans fichier : écran d'accueil, Ouvrir une vidéo)\n"
      "  --model=FICHIER.onnx   modèle SPAN ×2 (défaut : {})\n"
      "  --no-upscale           image source telle quelle\n"
      "  --window=LxH           taille de la fenêtre (défaut 1280x720)\n"
      "  --fullscreen           plein écran dès l'ouverture\n"
      "  --start=S              position de départ, en secondes (sinon, là où on s'était arrêté)\n"
      "  --no-resume            ni reprise ni mémoire de la position pour cette lecture\n"
      "  --seconds=S            quitte après S secondes de lecture\n"
      "  --fullscreen-after=S   passe en plein écran après S secondes\n"
      "  --stats=FICHIER        temps de chaque image présentée\n"
      "  --snapshot=FICHIER     la fenêtre en PPM, après --snapshot-at=S secondes (3) de lecture,\n"
      "                         ou de construction d'un moteur si elle dure jusque-là\n"
      "  --snapshot-frames=N    N présentations de suite (FICHIER-001.ppm…), pour le scintillement\n"
      "  --rife-image           RIFE image à chaque instant (sans la variante mouvement)\n"
      "  --sdr                  sortie SDR même sur un écran HDR (essais du grade SDR)\n"
      "  --one-lane             une seule file de calcul : plus lent, mais le profil GPU donne le\n"
      "                         coût propre de chaque moteur\n"
      "  --no-vrr               écran à fréquence fixe même si le VRR est armé (essai du mode Rapide)\n"
      "  --software-gpu         accepte un Vulkan logiciel (lavapipe), pour les essais sans GPU\n"
      "  --mute                 son coupé pour cette lecture\n"
      "  --data-dir=DOSSIER     réglages, profils et journal dans ce dossier (moteurs et modèles partagés)\n"
      "  --bench-level=N        joue la chaîne du niveau 0 à 4 (ou auto) ; --bench-out=FICHIER y écrit les "
      "mesures\n"
      "  --first-run            écran de bienvenue (moteurs, profil de démarrage), même déjà vu\n"
      "  --keys=S:touche,…      touches pressées d'office (essais), ex. 3:Right,5:Space ;\n"
      "                         export ou export=FICHIER.mkv (ou .mp4, .mov) lance l'export de\n"
      "                         la chaîne en cours\n"
      "  --work-height=N        réduit la source à N lignes avant SPAN (défaut 720 ; 0 jamais)\n"
      "  --rife=N               interpolation ×N avant SPAN (1.5, 2 à 10 ; défaut {} ; 0 aucune)\n"
      "  --rife-model=FICHIER   modèle RIFE (défaut : {})\n"
      "  --truehdr              RTX HDR (Maxine TrueHDR), SDK importé une fois depuis --truehdr-sdk\n"
      "  --truehdr-peak=NITS    pic de TrueHDR (défaut : celui de l'écran, sinon 1000)\n"
      "  --truehdr-contrast=N, --truehdr-saturation=N (0 à 200, 100), --truehdr-gray=N (10 à 100, 50)\n"
      "ajp --register           inscrit le lecteur auprès du système (« Ouvrir avec », lecteur par\n"
      "                         défaut) ; --unregister retire cette inscription\n"
      "Touches : espace pause, flèches seek 5 s, f plein écran, q quitter.\n",
      kDefaultModelFile, kDefaultRifeFactor, kDefaultRifeFile);
}

std::vector<std::filesystem::path> default_model_candidates(const std::filesystem::path& home,
                                                            const std::filesystem::path& bundled) {
  std::vector<std::filesystem::path> candidates{
      home / ".local/share/AnimeJaNai-HDR-linux/animejanai/onnx" / kDefaultModelFile,
      home /
          "Documents/mpv-AnimeJaNai-HDR/BuildMpvUpscale2xAnimeJaNai/mpv-upscale-2x_animejanai/animejanai/"
          "onnx" /
          kDefaultModelFile,
  };
  if (!bundled.empty()) candidates.push_back(bundled / "onnx" / kDefaultModelFile);
  return candidates;
}

bool valid_factor(Factor factor) {
  if (factor.den == 1) return limits::kRifeFactor.contains(factor.num);
  return factor.den == 2 && factor.num % 2 == 1 && factor.num >= 3 && factor.num < 2 * limits::kRifeFactor.hi;
}

std::optional<Factor> parse_factor(std::string_view text) {
  Factor factor;
  const std::size_t point = text.find_first_of(".,");
  if (point == std::string_view::npos) {
    const auto whole = number<int>(text);
    if (!whole) return std::nullopt;
    factor.num = *whole;
  } else {
    const auto whole = number<int>(text.substr(0, point));
    const std::string_view fraction = text.substr(point + 1);
    if (!whole || (fraction != "5" && fraction != "0")) return std::nullopt;
    factor = fraction == "5" ? Factor{.num = 2 * *whole + 1, .den = 2} : Factor{.num = *whole, .den = 1};
  }
  if (!valid_factor(factor)) return std::nullopt;
  return factor;
}

std::string factor_text(Factor factor) {
  if (factor.den == 2) return std::format("×{},5", factor.num / 2);
  return std::format("×{}", factor.num);
}

std::vector<std::filesystem::path> default_rife_candidates(const std::filesystem::path& home,
                                                           const std::filesystem::path& bundled) {
  const std::array files{kDefaultRifeFile};
  return default_rife_candidates(home, files, bundled);
}

std::vector<std::filesystem::path> default_rife_candidates(const std::filesystem::path& home,
                                                           std::span<const std::string_view> files,
                                                           const std::filesystem::path& bundled) {
  std::vector<std::filesystem::path> dirs{home / ".local/share/AnimeJaNai-HDR-linux/animejanai/rife",
                                          home / ".local/share/AnimeJaNai/animejanai/rife"};
  if (!bundled.empty()) dirs.push_back(bundled / "rife");
  std::vector<std::filesystem::path> candidates;
  for (const std::filesystem::path& dir : dirs) {
    for (const std::string_view file : files) candidates.push_back(dir / file);
  }
  return candidates;
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
