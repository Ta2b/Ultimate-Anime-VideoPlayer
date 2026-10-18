#include "core/debug_flags.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <format>
#include <utility>

#include "core/model_index.h"

namespace ajp::core {
namespace {

using Result = std::expected<void, Refusal>;

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

template <class E, std::size_t N>
std::optional<E> find_value(std::string_view text,
                            const std::array<std::pair<std::string_view, E>, N>& names) {
  for (const auto& [name, value] : names) {
    if (name == text) return value;
  }
  return std::nullopt;
}

constexpr std::array<std::pair<std::string_view, LogLevel>, 6> kLogLevels{{
    {"trace", LogLevel::Trace},
    {"debug", LogLevel::Debug},
    {"info", LogLevel::Info},
    {"warn", LogLevel::Warn},
    {"error", LogLevel::Error},
    {"off", LogLevel::Off},
}};
constexpr std::array<std::pair<std::string_view, RingFormat>, 2> kRingFormats{
    {{"rgba16", RingFormat::Rgba16Unorm}, {"yuv444p16", RingFormat::Yuv444p16}}};

// « 2560x1440@165 » ou « 2560x1440@165:hdr ».
std::optional<DisplayInfo> parse_display(std::string_view text) {
  DisplayInfo display;
  if (text.ends_with(":hdr")) {
    display.hdr = true;
    text.remove_suffix(4);
  }
  const std::size_t x = text.find('x');
  const std::size_t at = text.find('@');
  if (x == std::string_view::npos || at == std::string_view::npos || at < x) return std::nullopt;
  const auto width = number<std::int64_t>(text.substr(0, x));
  const auto height = number<std::int64_t>(text.substr(x + 1, at - x - 1));
  const auto hz = number<double>(text.substr(at + 1));
  if (!width || !height || !hz) return std::nullopt;
  if (!limits::kDisplaySide.contains(*width) || !limits::kDisplaySide.contains(*height) ||
      !limits::kRefreshHz.contains(*hz)) {
    return std::nullopt;
  }
  display.width = static_cast<int>(*width);
  display.height = static_cast<int>(*height);
  display.refresh_hz = *hz;
  return display;
}

Result bad_value(std::string_view name, std::string_view value) {
  return refuse(RefusalCode::FieldOutOfRange, std::format("--{}={}", name, value));
}

using Apply = Result (*)(DebugFlags& flags, std::string_view value);

struct FlagSpec {
  std::string_view name;   // sans « -- »
  std::string_view value;  // forme attendue de la valeur ; vide pour un simple drapeau
  std::string_view help;
  Apply apply;
};

constexpr std::array<FlagSpec, 10> kFlags{{
    {"log-level", "trace|debug|info|warn|error|off", "niveau du journal (info)",
     [](DebugFlags& f, std::string_view v) -> Result {
       const auto level = find_value(v, kLogLevels);
       if (!level) return bad_value("log-level", v);
       f.log_level = *level;
       return {};
     }},
    {"vulkan-validation", "", "couches de validation Vulkan",
     [](DebugFlags& f, std::string_view) -> Result {
       f.vulkan_validation = true;
       return {};
     }},
    {"no-cuda-graphs", "", "chaque paire lancée sans graphe CUDA",
     [](DebugFlags& f, std::string_view) -> Result {
       f.no_cuda_graphs = true;
       return {};
     }},
    {"trace-pairs", "", "temps et attentes de chaque paire au journal",
     [](DebugFlags& f, std::string_view) -> Result {
       f.trace_pairs = true;
       return {};
     }},
    {"no-exact-reuse", "", "recalcule chaque paire au lieu de la réutiliser",
     [](DebugFlags& f, std::string_view) -> Result {
       f.no_exact_reuse = true;
       return {};
     }},
    {"dump-plan", "", "écrit chaque Plan résolu au journal",
     [](DebugFlags& f, std::string_view) -> Result {
       f.dump_plan = true;
       return {};
     }},
    {"engine", "NOM", "impose ce moteur : vulkan, tensorrt, migraphx, openvino, directml, metal",
     [](DebugFlags& f, std::string_view v) -> Result {
       // Les SDK RTX Video et Maxine font TrueHDR, pas l'inférence : ils ne s'imposent pas ici.
       const auto engine = parse_engine(v);
       if (!engine || *engine == EngineId::RtxVideoSdk || *engine == EngineId::MaxineVfx) {
         return bad_value("engine", v);
       }
       f.force_engine = engine;
       return {};
     }},
    {"ring-format", "rgba16|yuv444p16", "impose le format du ring de sortie",
     [](DebugFlags& f, std::string_view v) -> Result {
       const auto format = find_value(v, kRingFormats);
       if (!format) return bad_value("ring-format", v);
       f.ring_format = format;
       return {};
     }},
    {"display", "LxH@Hz[:hdr]", "remplace l'écran détecté",
     [](DebugFlags& f, std::string_view v) -> Result {
       const auto display = parse_display(v);
       if (!display) return bad_value("display", v);
       f.display = *display;
       return {};
     }},
    {"max-frames", "N", "s'arrête après N images (0 : sans limite)",
     [](DebugFlags& f, std::string_view v) -> Result {
       const auto frames = number<std::int64_t>(v);
       if (!frames || !limits::kMaxFrames.contains(*frames)) return bad_value("max-frames", v);
       f.max_frames = *frames;
       return {};
     }},
}};

std::string usage_of(const FlagSpec& spec) {
  return spec.value.empty() ? std::format("--{}", spec.name) : std::format("--{}={}", spec.name, spec.value);
}

Result parse_option(std::string_view option, CommandLine& line) {
  const std::size_t equals = option.find('=');
  const std::string_view name = option.substr(0, equals);
  const bool has_value = equals != std::string_view::npos;
  const std::string_view value = has_value ? option.substr(equals + 1) : std::string_view{};

  if (name == "help" && !has_value) {
    line.help = true;
    return {};
  }
  const auto spec = std::ranges::find(kFlags, name, &FlagSpec::name);
  if (spec == kFlags.end()) return refuse(RefusalCode::OptionUnknown, std::format("--{}", name));
  if (spec->value.empty() && has_value) {
    return refuse(RefusalCode::FieldType, std::format("--{} ne prend pas de valeur", name));
  }
  if (!spec->value.empty() && !has_value) return refuse(RefusalCode::FieldMissing, usage_of(*spec));
  return spec->apply(line.debug, value);
}

}  // namespace

std::expected<CommandLine, Refusal> parse_command_line(std::span<const std::string_view> args) {
  CommandLine line;
  bool options_done = false;
  for (const std::string_view arg : args) {
    if (options_done || arg == "-" || !arg.starts_with('-')) {
      line.inputs.emplace_back(arg);
    } else if (arg == "--") {
      options_done = true;
    } else if (!arg.starts_with("--")) {
      return refuse(RefusalCode::OptionUnknown, std::string(arg));
    } else if (auto r = parse_option(arg.substr(2), line); !r) {
      return std::unexpected(r.error());
    }
  }
  return line;
}

std::expected<CommandLine, Refusal> parse_command_line(int argc, const char* const* argv) {
  std::vector<std::string_view> args;
  for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
  return parse_command_line(args);
}

std::string command_line_help() {
  std::size_t width = 0;
  for (const FlagSpec& spec : kFlags) width = std::max(width, usage_of(spec).size());
  std::string text = "Usage : ajp [options] [--] [fichier…]\n\nOptions de débogage :\n";
  for (const FlagSpec& spec : kFlags) text += std::format("  {:<{}}  {}\n", usage_of(spec), width, spec.help);
  return text + std::format("  {:<{}}  cette aide\n", "--help", width);
}

}  // namespace ajp::core
