#include "app/play/options.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <format>
#include <system_error>
#include <utility>

#include "app/play/i18n.h"
#include "app/play/settings.h"
#include "core/charconv.h"
#include "core/utf8_path.h"

namespace ajp::app::play {
namespace {

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
                  std::format("{}: \"{}\" is not a duration in seconds "
                              "between {} and {}",
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
                  std::format("--window: \"{}\" is not a WxH size, sides from {} to {}", text,
                              limits::kWindowSide.lo, limits::kWindowSide.hi));
  }
  options.window_width = *width;
  options.window_height = *height;
  return {};
}

// Les options à valeur, dans l'ordre où elles se cherchent (--snapshot-at avant --snapshot).
constexpr std::array<std::string_view, 23> kValuedOptions{
    "--model",       "--window",          "--start",           "--seconds",          "--fullscreen-after",
    "--stats",       "--snapshot-at",     "--snapshot-frames", "--snapshot",         "--keys",
    "--work-height", "--rife-model",      "--anime-contrast",  "--anime-saturation", "--anime-highlights",
    "--rife",        "--truehdr-peak",    "--data-dir",        "--bake-looks",       "--bench-level",
    "--bench-out",   "--rife-calm-model", "--motion"};

// Les options qui ne donnent qu'un chemin.
using PathMember = std::optional<std::filesystem::path> PlayOptions::*;
std::optional<PathMember> path_option(std::string_view name) {
  constexpr std::array<std::pair<std::string_view, PathMember>, 8> kPaths{{
      {"--model", &PlayOptions::model},
      {"--data-dir", &PlayOptions::data_dir},
      {"--bake-looks", &PlayOptions::bake_looks},
      {"--bench-out", &PlayOptions::bench_out},
      {"--stats", &PlayOptions::stats},
      {"--snapshot", &PlayOptions::snapshot},
      {"--rife-model", &PlayOptions::rife_model},
      {"--rife-calm-model", &PlayOptions::rife_calm_model},
  }};
  for (const auto& [option, member] : kPaths)
    if (option == name) return member;
  return std::nullopt;
}

std::expected<void, Refusal> motion(std::string_view text, PlayOptions& options) {
  if (text == "rife") {
    options.motion = MotionChoice::Rife;
  } else if (text == "drba") {
    options.motion = MotionChoice::DistilDrba;
  } else {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("--motion: \"{}\" is neither rife nor drba", text));
  }
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
      return refuse(RefusalCode::FieldOutOfRange, std::format("--keys: \"{}\" is not S:key", item));
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
  if (i + 1 >= args.size()) return refuse(RefusalCode::FieldMissing, std::format("{}: missing value", name));
  return args[++i];
}

// Les options d'essai (P2.35) : bascules des chantiers et mesures, acceptées avec --debug seulement.
constexpr std::array<std::string_view, 29> kTrialOptions{
    "--rife-image",   "--ring-rgb10a2",    "--no-ring-rgb10a2", "--hdr-metadata",      "--no-hdr-metadata",
    "--probe-nvdec",  "--no-probe-nvdec",  "--shot-cuts",       "--no-shot-cuts",      "--one-lane",
    "--skip-held",    "--anime-shots",     "--no-anime-shots",  "--fast-steady",       "--smooth-drawings",
    "--ring-watcher", "--no-ring-watcher", "--precise-wait",    "--no-precise-wait",   "--audio-pll",
    "--no-audio-pll", "--vsync-index",     "--no-vsync-index",  "--present",           "--no-present",
    "--no-vrr",       "--rife-calm-model", "--rife-warm",       "--profile-dispatches"};

bool is_trial(std::string_view arg) {
  const std::string_view name = arg.substr(0, arg.find('='));
  return std::ranges::find(kTrialOptions, name) != kTrialOptions.end();
}

bool asks_debug(std::span<const std::string_view> args) {
  for (const std::string_view arg : args) {
    if (arg == "--") return false;
    if (arg == "--debug") return true;
  }
  return false;
}

struct Flag {
  std::string_view name;
  bool PlayOptions::* field;
  bool value;
};

// Les drapeaux d'essai sans valeur : leur champ de PlayOptions et ce qu'ils y posent.
constexpr std::array<Flag, 15> kFlags{{
    {"--no-vrr", &PlayOptions::no_vrr, true},
    {"--one-lane", &PlayOptions::one_lane, true},
    {"--profile-dispatches", &PlayOptions::profile_dispatches, true},
    {"--rife-warm", &PlayOptions::rife_warm, true},
    {"--skip-held", &PlayOptions::skip_held, true},
    {"--fast-steady", &PlayOptions::fast_steady, true},
    {"--smooth-drawings", &PlayOptions::smooth_drawings, true},
    {"--ring-watcher", &PlayOptions::ring_watcher, true},
    {"--no-ring-watcher", &PlayOptions::ring_watcher, false},
    {"--precise-wait", &PlayOptions::precise_wait, true},
    {"--no-precise-wait", &PlayOptions::precise_wait, false},
    {"--audio-pll", &PlayOptions::audio_pll, true},
    {"--no-audio-pll", &PlayOptions::audio_pll, false},
    {"--vsync-index", &PlayOptions::vsync_index, true},
    {"--no-vsync-index", &PlayOptions::vsync_index, false},
}};

bool set_flag(std::string_view arg, PlayOptions& options) {
  const auto flag = std::ranges::find(kFlags, arg, &Flag::name);
  if (flag == kFlags.end()) return false;
  options.*(flag->field) = flag->value;
  return true;
}

}  // namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity,readability-function-size)
std::expected<PlayOptions, Refusal> parse_play_options(std::span<const std::string_view> args) {
  PlayOptions options;
  options.debug = asks_debug(args);
  bool only_files = false;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string_view arg = args[i];
    if (!only_files && !options.debug && is_trial(arg))
      return refuse(RefusalCode::OptionUnknown, std::format("{}: trial option, give --debug too", arg));
    if (only_files || !arg.starts_with("--")) {
      if (!options.file.empty()) {
        return refuse(RefusalCode::OptionUnknown,
                      std::format("one file at a time: \"{}\" is one too many", arg));
      }
      options.file = std::filesystem::path(arg);
      continue;
    }
    if (arg == "--") {
      only_files = true;
    } else if (arg == "--help") {
      options.help = true;
    } else if (arg == "--debug" || set_flag(arg, options)) {
      // --debug est lu avant la boucle (asks_debug) ; set_flag pose un drapeau d'essai de kFlags
    } else if (arg == "--no-upscale") {
      options.upscale = false;
      options.given.emplace_back(arg);
    } else if (arg == "--no-restore-lines") {  // l'essai B de « Restaurer les traits » (ajp compare)
      options.given.emplace_back(arg);
    } else if (arg == "--fullscreen") {
      options.fullscreen = true;
    } else if (arg == "--hidden") {
      options.hidden = true;
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
    } else if (arg == "--ring-rgb10a2") {
      options.ring_rgb10a2 = true;
    } else if (arg == "--no-ring-rgb10a2") {
      options.ring_rgb10a2 = false;
    } else if (arg == "--probe-nvdec") {
      options.probe_nvdec = true;
    } else if (arg == "--no-probe-nvdec") {
      options.probe_nvdec = false;
    } else if (arg == "--shot-cuts") {
      options.shot_cuts = true;
    } else if (arg == "--no-shot-cuts") {
      options.shot_cuts = false;  // les coupes de l'interpolation mesurées par RIFE seul
    } else if (arg == "--anime-shots") {
      options.anime_shots = true;
    } else if (arg == "--no-anime-shots") {
      options.anime_shots = false;  // le relèvement fixe, la lumière de la scène mesurée en direct
    } else if (arg == "--no-hdr-metadata" || arg == "--hdr-metadata=off") {
      options.kde_sdr_nits.reset();
      options.kde_sdr_auto = false;
    } else if (arg == "--hdr-metadata" || arg.starts_with("--hdr-metadata=")) {
      options.kde_sdr_nits = kDefaultKdeSdrNits;
      options.kde_sdr_auto = arg == "--hdr-metadata";
      if (arg.size() > std::string_view("--hdr-metadata").size()) {
        const std::string_view text = arg.substr(std::string_view("--hdr-metadata=").size());
        const auto nits = number<int>(text);
        if (!nits || *nits < 50 || *nits > 1000)
          return refuse(
              RefusalCode::FieldOutOfRange,
              std::format("--hdr-metadata: \"{}\" is not an SDR brightness from 50 to 1000 nits", text));
        options.kde_sdr_nits = static_cast<double>(*nits);
      }
    } else if (arg.starts_with("--present=")) {
      const auto wanted = parse_present(arg.substr(std::string_view("--present=").size()));
      if (!wanted)
        return refuse(RefusalCode::FieldOutOfRange,
                      std::format("{}: auto, timing, wait, wayland or open", arg));
      options.present = *wanted;
    } else if (arg == "--no-present") {
      options.present = PresentWanted::Open;
    } else if (arg == "--no-resume") {
      options.resume = false;
    } else if (arg == "--register" || arg == "--unregister") {
      const Registration wanted = arg == "--register" ? Registration::Register : Registration::Unregister;
      if (options.registration != Registration::None && options.registration != wanted)
        return refuse(RefusalCode::OptionUnknown, "--register and --unregister together");
      options.registration = wanted;
    } else {
      std::expected<void, Refusal> applied;
      bool known = false;
      for (const std::string_view name : kValuedOptions) {
        auto value = valued(args, i, name);
        if (!value) return std::unexpected(std::move(value.error()));
        if (!*value) continue;
        known = true;
        options.given.emplace_back(name);
        const std::string_view text = **value;
        if (const auto member = path_option(name)) {
          options.*(*member) = std::filesystem::path(text);
        } else if (name == "--motion") {
          applied = motion(text, options);
        } else if (name == "--window") {
          applied = window(text, options);
        } else if (name == "--bench-level") {
          const auto level = text == "auto" ? std::optional<int>(-1) : number<int>(text);
          if (!level || *level < -1 || *level >= 5)
            return refuse(RefusalCode::FieldOutOfRange,
                          std::format("--bench-level: \"{}\" is neither auto nor 0 to 4", text));
          options.bench_level = level;
        } else if (name == "--keys") {
          applied = keys(text, options);
        } else if (name == "--snapshot-frames") {
          const auto frames = number<int>(text);
          if (!frames || *frames < 1 || *frames > kMaxSnapshotFrames)
            return refuse(RefusalCode::FieldOutOfRange,
                          std::format("--snapshot-frames: \"{}\" outside 1 to {}", text, kMaxSnapshotFrames));
          options.snapshot_frames = *frames;
        } else if (name == "--truehdr-peak") {
          // Les bornes de play.toml (limits::), 0 pour « celui de l'écran ».
          const auto setting = number<int>(text);
          const core::Range<int> range = limits::kTrueHdrPeak;
          if (!setting || !(range.contains(*setting) || *setting == 0)) {
            return refuse(RefusalCode::FieldOutOfRange,
                          std::format("{}: \"{}\" outside {} to {}", name, text, range.lo, range.hi));
          }
          options.true_hdr_peak = *setting;
        } else if (name == "--anime-contrast" || name == "--anime-saturation" ||
                   name == "--anime-highlights") {
          const auto percent = number<int>(text);
          if (!percent || *percent < 0 || *percent > 200) {
            return refuse(RefusalCode::FieldOutOfRange,
                          std::format("{}: \"{}\" is not a percentage between 0 and 200", name, text));
          }
          if (name == "--anime-contrast") options.anime_contrast = *percent;
          if (name == "--anime-saturation") options.anime_saturation = *percent;
          if (name == "--anime-highlights") options.anime_highlights = *percent;
        } else if (name == "--rife") {
          const auto factor = parse_factor(text);
          if (!factor) {
            return refuse(RefusalCode::FieldOutOfRange,
                          std::format("--rife: \"{}\" is not a factor between {} and {} (whole or half)",
                                      text, limits::kRifeFactor.lo, limits::kRifeFactor.hi));
          }
          options.rife_factor = factor->num;
          options.rife_factor_den = factor->den;
        } else if (name == "--work-height") {
          const auto lines = number<int>(text);
          if (!lines || !limits::kWorkHeight.contains(*lines)) {
            return refuse(RefusalCode::FieldOutOfRange,
                          std::format("--work-height: \"{}\" is not a line count between {} and {}", text,
                                      limits::kWorkHeight.lo, limits::kWorkHeight.hi));
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
      if (!known) return refuse(RefusalCode::OptionUnknown, std::format("unknown option: {}", arg));
    }
  }
  if (options.registration != Registration::None && !options.file.empty()) {
    return refuse(RefusalCode::OptionUnknown, "--register and --unregister take no video file");
  }
  return options;  // sans fichier : l'écran d'accueil (home_screen.h)
}

bool PlayOptions::was_given(std::string_view name) const {
  return std::ranges::find(given, name) != given.end();
}

bool span_runs(bool upscale, bool restore_lines, int source_height, int work_height) {
  if (!upscale) return false;
  return restore_lines || work_height <= 0 || source_height <= work_height;
}

WorkSize work_size(int width, int height, int work_height) {
  if (work_height <= 0 || height <= work_height || width <= 0 || height <= 0) return {width, height};
  const auto scaled = static_cast<int>(std::lround(static_cast<double>(width) * work_height / height));
  return {std::max(2, scaled + (scaled % 2)), work_height};
}

WorkSize chain_work_size(bool upscale, bool restore_lines, int width, int height, int work_height) {
  const bool kept = upscale && !span_runs(upscale, restore_lines, height, work_height);
  return work_size(width, height, kept ? 0 : work_height);
}

std::expected<BenchOptions, Refusal> parse_bench_options(std::span<const std::string_view> args) {
  BenchOptions options;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string_view arg = args[i];
    if (!arg.starts_with("--")) {
      if (!options.file.empty())
        return refuse(RefusalCode::OptionUnknown,
                      std::format("one file at a time: \"{}\" is one too many", arg));
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
                      std::format("--seconds: \"{}\" is not a duration from 5 to 120 s", **length));
      options.seconds = *value;
    } else {
      return refuse(RefusalCode::OptionUnknown, std::format("unknown option: {}", arg));
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
                      std::format("one file at a time: \"{}\" is one too many", arg));
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
                        std::format("--seconds: \"{}\" is not a duration from 5 to 120 s", text));
        options.seconds = *value_s;
      }
      break;
    }
    if (!known) return refuse(RefusalCode::OptionUnknown, std::format("unknown option: {}", arg));
  }
  if (!options.help && options.file.empty())
    return refuse(RefusalCode::FieldMissing, "ajp compare: the video to play is missing");
  return options;
}

// L'aide de la ligne de commande, en anglais comme l'interface par défaut : elle s'écrit avant que les
// réglages (et la langue choisie) ne soient lus.
std::string compare_usage() {
  return "ajp compare <file> --a=\"OPTIONS\" --b=\"OPTIONS\" [options]   two chains side by side, sound off\n"
         "  --a, --b            \"ajp play\" options of each variant, e.g. --a=\"--rife=3\" "
         "--b=\"--rife=4\"\n"
         "  --a-settings=FILE, --b-settings=FILE   starting play.toml (DRBA poses, interpolation engine…)\n"
         "  --seconds=S         length of one run (5 to 120, default 12); A, B, B, A: the worse of two "
         "passes\n"
         "  --report=FILE       the report, besides standard output; --windowed: in a window\n";
}

std::string bench_usage() {
  return "ajp bench [file] [options]   rates the machine at each level, sound off, hidden window; without a "
         "file, the player's test sequence\n"
         "  --seconds=S      length of one run (5 to 120, default 12)\n"
         "  --report=FILE    the report, besides standard output\n"
         "  --windowed       plays in a window instead of full screen\n"
         "  --visible        shows the runs' window (hidden by default)\n"
         "The result is kept in bench.toml (card, driver, level, date) and replaces the card's estimate.\n";
}

std::string play_usage(bool debug) {
  const std::string usage = std::format(
      "ajp play [file] [options]   (without a file: home screen, Open a video)\n"
      "  --model=FILE.onnx      SPAN ×2 model (default: {})\n"
      "  --no-upscale           source picture, reduced only to the working resolution\n"
      "  --no-restore-lines     a source above the working resolution keeps its detail, no SPAN\n"
      "  --window=WxH           window size (default 1280x720)\n"
      "  --fullscreen           full screen from the start\n"
      "  --start=S              start position, in seconds (otherwise, where playback stopped)\n"
      "  --no-resume            neither resume nor remember the position for this playback\n"
      "  --seconds=S            quits after S seconds of playback, from the first frame shown\n"
      "  --fullscreen-after=S   goes full screen after S seconds\n"
      "  --stats=FILE           timing of each presented frame\n"
      "  --snapshot=FILE        the window as PPM, after --snapshot-at=S seconds (3) of playback,\n"
      "                         or of an engine build if it lasts that long; on an HDR10\n"
      "                         output, only a .pq.ppm name keeps the graded picture\n"
      "  --snapshot-frames=N    N presentations in a row (FILE-001.ppm…), for flicker\n"
      "  --sdr                  SDR output even on an HDR screen (SDR grade tests)\n"
      "  --bake-looks=FILE.cube writes play.toml's LUT stack (strengths included) as a single\n"
      "                         65-point LUT, then stops\n"
      "  --anime-contrast=N, --anime-saturation=N, --anime-highlights=N\n"
      "                         the anime grade's punch: contrast, saturation, highlights, 0 to 200% (100: "
      "as is)\n"
      "  --software-gpu         accepts a software Vulkan (lavapipe), for tests without a GPU\n"
      "  --mute                 sound off for this playback\n"
      "  --data-dir=FOLDER      settings and log in this folder (engines and models shared)\n"
      "  --bench-level=N        plays the chain of level 0 to 4 (or auto); --bench-out=FILE writes the\n"
      "                         measurements there\n"
      "  --first-run            welcome screen (engines), even if already seen\n"
      "  --keys=S:key,…         keys pressed automatically (tests), e.g. 3:Right,5:Space;\n"
      "                         export or export=FILE.mkv (or .mp4, .mov) starts exporting\n"
      "                         the current chain\n"
      "  --work-height=N        shrinks the source to N lines before SPAN (default: the preset's; 0 never)\n"
      "  --rife=N               ×N interpolation (1.5, 2 to 10; default: the preset's; 0 none); order:\n"
      "                         Warp and SPAN, Auto by default\n"
      "  --rife-model=FILE      RIFE model (default: the preset's, {} without one)\n"
      "  --motion=rife|drba     interpolation engine, RIFE or DistilDRBA (default: the preset's)\n"
      "  --truehdr-peak=NITS    peak aimed at by the HDR grade (default: the screen's, else 1000)\n"
      "ajp --register           registers the player with the system (\"Open with\", default\n"
      "                         player); --unregister removes this registration\n"
      "Keys: space pause, ← → seek 5 s, ↑ ↓ volume, m mute, f full screen (Esc leaves it), q quit,\n"
      "  a next audio track, j next subtitles, n / Shift+n next / previous episode,\n"
      "  Page↑ / Page↓ next / previous chapter, o skip the credits, . and , frame by frame, z / x\n"
      "  subtitle delay, Ctrl+- / Ctrl++ audio delay, p settings, Tab statistics, Ctrl+0 no processing,\n"
      "  Ctrl+G grade, Ctrl+I interpolation.\n",
      kDefaultModelFile, kDefaultRifeFile);
  if (!debug) return usage + "Trial options: ajp --debug --help\n";
  return usage +
         "Trial options, with --debug:\n"
         "  --rife-image           image RIFE at every instant (without the motion variant)\n"
         "  --ring-rgb10a2         RGB10A2 ring under TensorRT (half the memory, banding not\n"
         "                         measured yet); default and --no-ring-rgb10a2: RGBA16\n"
         "  --hdr-metadata[=NITS]  HDR10 output with its metadata, KWin's compression done by the\n"
         "                         grade (NITS: KDE's SDR brightness, read from KDE without a value);\n"
         "                         default under KDE and Windows; --no-hdr-metadata: compression\n"
         "                         left to KWin (Windows: clipping at the screen's peak)\n"
         "  --probe-nvdec          shot pre-analysis decoded by NVDEC at 256x144 (default;\n"
         "                         --no-probe-nvdec: software decoding)\n"
         "  --no-shot-cuts         interpolation cuts measured by RIFE alone (default: the pre-analysis\n"
         "                         shots decide, RIFE when they do not cover the pair)\n"
         "  --one-lane             a single compute queue: slower, but the GPU profile gives each\n"
         "                         engine's own cost\n"
         "  --profile-dispatches   Vulkan engine: GPU time and dispatch count per shader of each network,\n"
         "                         logged when the network is unloaded\n"
         "  --skip-held            playback: an instant identical to the shown frame is neither computed "
         "nor\n"
         "                         written (trial; off by default)\n"
         "  --anime-shots          SDR output only: the grade decided per shot (default); on HDR output the "
         "lift is\n"
         "                         fixed whatever this says; --no-anime-shots: fixed lift on SDR too\n"
         "  --fast-steady          trial: Instant mode at a constant step, N frames per held source in a "
         "shot\n"
         "  --smooth-drawings      trial: a drawing held two frames blends into motion (motion RIFE)\n"
         "  --rife-calm-model=FILE trial: a lighter RIFE for the middle pass of calm pairs (motion RIFE)\n"
         "  --rife-warm            trial: calm pairs reuse the coarse RIFE blocks of the previous pair;\n"
         "                         trades quality for speed, neither measured (ghosting if a calm shot\n"
         "                         speeds up)\n"
         "  --ring-watcher         a thread waits for ring writes and wakes the display (default;\n"
         "                         --no-ring-watcher: the display polls)\n"
         "  --precise-wait         waits for the next frame to the microsecond (default; --no-precise-wait:\n"
         "                         to the millisecond)\n"
         "  --audio-pll            audio clock filtered by a phase-locked loop (default;\n"
         "                         --no-audio-pll: the raw clock)\n"
         "  --vsync-index          frame chosen by its measured vblank number, with hysteresis (default;\n"
         "                         --no-vsync-index: the clock)\n"
         "  --present=LEVEL        presentation feedback wait (default), auto, timing, wayland or open;\n"
         "                         --no-present: open, the former presentation\n"
         "  --no-vrr               fixed refresh screen even if VRR is armed (Instant mode trial)\n";
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
  if (factor.den == 2) return std::format("×{}{}5", factor.num / 2, i18n::decimal_point());
  if (factor.den <= 1 || factor.num % factor.den == 0)
    return std::format("×{}", factor.num / std::max(factor.den, 1));
  // Un facteur de l'Auto (palier fin ou fraction exacte d'un écran fixe) : deux décimales, « ×3,44 ».
  const long centi = std::lround(100.0 * factor.num / factor.den);
  return std::format("×{}{}{:02}", centi / 100, i18n::decimal_point(), centi % 100);
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

// Le fichier donné par l'option (il doit exister), sinon le premier candidat présent. Un nom seul, sans
// dossier (« rife_v4.25.onnx »), absent du dossier courant, se cherche dans les dossiers des candidats
// (A18-18) : le banc et la ligne de commande nomment un modèle du catalogue sans savoir où il est installé.
std::expected<std::filesystem::path, Refusal> find_file(const std::optional<std::filesystem::path>& given,
                                                        std::string_view option, std::string_view what,
                                                        std::span<const std::filesystem::path> candidates) {
  std::error_code error;
  if (given) {
    if (std::filesystem::is_regular_file(*given, error)) return *given;
    if (!given->has_parent_path()) {
      for (const auto& path : candidates) {
        std::filesystem::path beside = path.parent_path() / *given;
        if (std::filesystem::is_regular_file(beside, error)) return beside;
      }
    }
    return refuse(RefusalCode::ModelUnreadable,
                  std::format("{}: {} not found", option, core::to_utf8(*given)));
  }
  std::string tried;
  for (const auto& path : candidates) {
    if (std::filesystem::is_regular_file(path, error)) return path;
    tried += std::format("\n  {}", path.string());
  }
  return refuse(RefusalCode::ModelUnreadable, std::format("default {} not found:{}", what, tried));
}

}  // namespace

std::expected<std::filesystem::path, Refusal> find_model(const PlayOptions& options,
                                                         std::span<const std::filesystem::path> candidates) {
  return find_file(options.model, "--model", "AnimeJaNai model", candidates);
}

std::expected<std::filesystem::path, Refusal> find_rife_model(
    const PlayOptions& options, std::span<const std::filesystem::path> candidates) {
  return find_file(options.rife_model, "--rife-model", "RIFE model", candidates);
}

}  // namespace ajp::app::play
