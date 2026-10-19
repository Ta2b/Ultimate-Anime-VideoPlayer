#include "app/cli.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <format>
#include <utility>

#include "app/cli_spec.h"
#include "core/model_index.h"
#include "core/profile.h"

namespace ajp::app {
namespace {

using cli::bit;
using cli::CommandOptions;
using cli::CommandSpec;
using cli::OptionSpec;
using Result = std::expected<void, Refusal>;

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

constexpr std::array<std::pair<std::string_view, ExportEncoder>, 4> kEncoders{{
    {"nvenc-hevc", ExportEncoder::NvencHevc},
    {"nvenc-av1", ExportEncoder::NvencAv1},
    {"svt-av1", ExportEncoder::SvtAv1},
    {"videotoolbox-hevc", ExportEncoder::VideoToolboxHevc},
}};

constexpr std::array<CommandSpec, 7> kCommands{{
    {CommandKind::Play, "play", "play <fichier>", "lit une vidéo",
     "Ouvre la vidéo dans la fenêtre de lecture, avec le profil de démarrage ou celui de --profile.",
     "ajp play episode01.mkv"},
    {CommandKind::Export, "export", "export <fichier> <sortie>",
     "traite une vidéo et l'écrit dans un fichier, sans fenêtre",
     "Exécute le même Plan que la lecture, sans horloge, et encode le résultat. L'audio, les "
     "chapitres et les sous-titres sont recopiés, sauf avec --burn-subtitles.",
     "ajp export episode01.mkv episode01-4k.mkv --profile=2"},
    {CommandKind::Bench, "bench", "bench <fichier>", "mesure le temps par paire et les attentes de l'hôte",
     "Joue la vidéo sans l'afficher et mesure le temps par paire et host_waits. Avec --dump, écrit "
     "le framemd5 des images produites, à comparer à celui de l'export.",
     "ajp bench episode01.mkv --max-frames=2000"},
    {CommandKind::ImportSdk, "import-sdk", "import-sdk <dossier>",
     "importe le SDK TrueHDR (Maxine VFX sous Linux, RTX Video sous Windows)",
     "Vérifie le dossier d'un SDK NVIDIA déjà extrait (fichiers, architecture, version), le copie "
     "dans le dossier des données et le choisit pour RTX HDR. Le SDK précédent reste en place si "
     "l'import est refusé.",
     "ajp import-sdk /usr/local/VideoFX"},
    {CommandKind::Engines, "engines", "engines [install [moteur…]]",
     "liste les moteurs et leur état ; « install » installe la sélection",
     "Sans argument : le GPU détecté, puis chaque moteur avec ses badges, son état et sa taille. "
     "« install » télécharge, vérifie et installe les moteurs cochés (au premier lancement, la "
     "présélection pour ce GPU) et les modèles des profils. Un moteur nommé (tensorrt, migraphx, "
     "openvino, directml) s'y ajoute, même s'il ne convient pas au GPU détecté.",
     "ajp engines install tensorrt"},
    {CommandKind::Help, "help", "help [commande]", "cette aide, ou celle d'une commande",
     "Affiche l'aide générale, ou celle de la commande nommée.", "ajp help export"},
    {CommandKind::Version, "version", "version", "version du lecteur", "Affiche la version du lecteur.",
     "ajp version"},
}};

// spec_of() et kind_of() indexent par CommandKind : les deux ordres doivent le suivre.
constexpr bool commands_in_kind_order() {
  for (std::size_t i = 0; i < kCommands.size(); ++i) {
    if (kCommands[i].kind != static_cast<CommandKind>(i)) return false;
  }
  return true;
}
static_assert(commands_in_kind_order());
static_assert(
    std::is_same_v<std::variant_alternative_t<static_cast<std::size_t>(CommandKind::Engines), CliCommand>,
                   EnginesArgs>);
static_assert(
    std::is_same_v<std::variant_alternative_t<static_cast<std::size_t>(CommandKind::Version), CliCommand>,
                   VersionArgs>);
static_assert(std::variant_size_v<CliCommand> == kCommands.size());

std::optional<int> profile_slot(std::string_view text) {
  int value = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size()) return std::nullopt;
  if (!core::limits::kStartProfile.contains(value)) return std::nullopt;
  return value;
}

constexpr std::uint8_t kProcessing =
    bit(CommandKind::Play) | bit(CommandKind::Export) | bit(CommandKind::Bench);

constexpr std::array<OptionSpec, 5> kOptions{{
    {"profile", "1..9", "profil à utiliser au lieu du profil de démarrage", kProcessing,
     [](CommandOptions& o, std::string_view v) -> Result {
       o.profile = profile_slot(v);
       if (!o.profile) return refuse(RefusalCode::FieldOutOfRange, std::format("--profile={} (1 à 9)", v));
       return {};
     }},
    {"encoder", "nvenc-hevc|nvenc-av1|svt-av1|videotoolbox-hevc", "encodeur (défaut : celui du GPU)",
     bit(CommandKind::Export),
     [](CommandOptions& o, std::string_view v) -> Result {
       const auto found = std::ranges::find(kEncoders, v, &std::pair<std::string_view, ExportEncoder>::first);
       if (found == kEncoders.end())
         return refuse(RefusalCode::FieldOutOfRange, std::format("--encoder={}", v));
       o.encoder = found->second;
       return {};
     }},
    {"burn-subtitles", "", "grave les sous-titres dans l'image", bit(CommandKind::Export),
     [](CommandOptions& o, std::string_view) -> Result {
       o.burn_subtitles = true;
       return {};
     }},
    {"overwrite", "", "remplace la sortie si elle existe", bit(CommandKind::Export),
     [](CommandOptions& o, std::string_view) -> Result {
       o.overwrite = true;
       return {};
     }},
    {"dump", "FICHIER", "écrit le framemd5 des images produites", bit(CommandKind::Bench),
     [](CommandOptions& o, std::string_view v) -> Result {
       o.dump = std::filesystem::path(std::string(v));
       return {};
     }},
}};

// Une option propre à une commande, gardée jusqu'à ce que la commande soit connue.
struct Option {
  const OptionSpec* spec;
  std::string_view value;
  bool has_value;
};

struct Scan {
  std::vector<std::string_view> positionals;
  std::vector<std::string_view> debug;  // transmises telles quelles à core::parse_command_line
  std::vector<Option> options;
  std::optional<std::filesystem::path> data_dir;
  bool help = false;
  bool version = false;
};

Result scan_option(std::string_view arg, Scan& scan) {
  const std::size_t equals = arg.find('=');
  const bool has_value = equals != std::string_view::npos;
  const std::string_view name = arg.substr(2, has_value ? equals - 2 : std::string_view::npos);
  const std::string_view value = has_value ? arg.substr(equals + 1) : std::string_view{};

  if (name == "help" || name == "version") {
    if (has_value) return refuse(RefusalCode::FieldType, std::format("--{} ne prend pas de valeur", name));
    (name == "help" ? scan.help : scan.version) = true;
  } else if (name == "data-dir") {
    if (value.empty()) return refuse(RefusalCode::FieldMissing, "--data-dir=DOSSIER");
    scan.data_dir = std::filesystem::path(std::string(value));
  } else if (const auto spec = std::ranges::find(kOptions, name, &OptionSpec::name); spec != kOptions.end()) {
    scan.options.push_back(Option{.spec = &*spec, .value = value, .has_value = has_value});
  } else {
    scan.debug.push_back(arg);  // core refuse lui-même une option inconnue
  }
  return {};
}

std::expected<Scan, Refusal> scan_arguments(std::span<const std::string_view> args) {
  Scan scan;
  bool options_done = false;
  for (const std::string_view arg : args) {
    if (options_done || arg == "-" || !arg.starts_with('-')) {
      scan.positionals.push_back(arg);
    } else if (arg == "--") {
      options_done = true;
    } else if (arg == "-h") {
      scan.help = true;
    } else if (!arg.starts_with("--")) {
      return refuse(RefusalCode::OptionUnknown,
                    std::format("{} : les options s'écrivent --nom ou --nom=valeur (voir ajp --help)", arg));
    } else if (auto r = scan_option(arg, scan); !r) {
      return std::unexpected(r.error());
    }
  }
  return scan;
}

// Distance d'édition, pour proposer la commande voisine d'une faute de frappe.
std::size_t edit_distance(std::string_view a, std::string_view b) {
  std::vector<std::size_t> row(b.size() + 1);
  for (std::size_t j = 0; j < row.size(); ++j) row[j] = j;
  for (std::size_t i = 1; i <= a.size(); ++i) {
    std::size_t diagonal = row[0];
    row[0] = i;
    for (std::size_t j = 1; j <= b.size(); ++j) {
      const std::size_t above = row[j];
      row[j] = std::min({row[j] + 1, row[j - 1] + 1, diagonal + (a[i - 1] == b[j - 1] ? 0 : 1)});
      diagonal = above;
    }
  }
  return row[b.size()];
}

std::optional<CommandKind> find_command(std::string_view word) {
  const auto found = std::ranges::find(kCommands, word, &CommandSpec::name);
  if (found == kCommands.end()) return std::nullopt;
  return found->kind;
}

std::unexpected<Refusal> unknown_command(std::string_view word) {
  std::string detail = std::format("commande « {} » inconnue", word);
  for (const CommandSpec& spec : kCommands) {
    if (edit_distance(word, spec.name) <= 2) {
      detail += std::format(" ; vouliez-vous dire « {} » ?", spec.name);
      return refuse(RefusalCode::OptionUnknown, detail);
    }
  }
  return refuse(
      RefusalCode::OptionUnknown,
      detail + std::format(" (voir ajp --help ; pour lire un fichier de ce nom : ajp play {})", word));
}

// « episode01.mkv », « ./film », « C:\x » : un mot qui n'est pas une commande mais un chemin.
bool looks_like_path(std::string_view word) {
  return word.find_first_of("./\\:") != std::string_view::npos;
}

std::expected<void, Refusal> expect_count(CommandKind kind, std::span<const std::string_view> args,
                                          std::size_t count) {
  const CommandSpec& spec = cli::spec_of(kind);
  if (args.size() < count) return refuse(RefusalCode::FieldMissing, std::format("ajp {}", spec.usage));
  if (args.size() > count) {
    return refuse(RefusalCode::OptionUnknown,
                  std::format("argument en trop « {} » ; usage : ajp {}", args[count], spec.usage));
  }
  return {};
}

std::filesystem::path path_of(std::string_view text) {
  return std::filesystem::path(std::string(text));
}

std::expected<core::EngineId, Refusal> installable_engine(std::string_view name) {
  const std::optional<core::EngineId> engine = core::parse_engine(name);
  if (!engine) {
    return refuse(
        RefusalCode::OptionUnknown,
        std::format("moteur « {} » inconnu ; moteurs à installer : tensorrt, migraphx, openvino, directml",
                    name));
  }
  switch (*engine) {
    case core::EngineId::Vulkan:
    case core::EngineId::Metal:
      return refuse(RefusalCode::FieldOutOfRange, std::format("{} est livré avec le lecteur", name));
    case core::EngineId::RtxVideoSdk:
    case core::EngineId::MaxineVfx:
      return refuse(RefusalCode::FieldOutOfRange,
                    std::format("{} s'importe depuis son dossier : ajp import-sdk <dossier>", name));
    case core::EngineId::TensorRt:
    case core::EngineId::MiGraphX:
    case core::EngineId::OpenVino:
    case core::EngineId::DirectMl:
      break;
  }
  return *engine;
}

std::expected<CliCommand, Refusal> engines_command(std::span<const std::string_view> args) {
  if (args.empty()) return EnginesArgs{};
  if (args.front() != "install") {
    return refuse(
        RefusalCode::OptionUnknown,
        std::format("engines : « {} » inconnu ; usage : ajp engines [install [moteur…]]", args.front()));
  }
  EnginesArgs engines{.install = true};
  for (const std::string_view name : args.subspan(1)) {
    auto engine = installable_engine(name);
    if (!engine) return std::unexpected(engine.error());
    if (std::ranges::contains(engines.engines, *engine)) {
      return refuse(RefusalCode::FieldOutOfRange, std::format("moteur « {} » nommé deux fois", name));
    }
    engines.engines.push_back(*engine);
  }
  return engines;
}

std::expected<CliCommand, Refusal> build_command(CommandKind kind, std::span<const std::string_view> args,
                                                 const CommandOptions& options) {
  const auto counted = [&](std::size_t count) { return expect_count(kind, args, count); };
  switch (kind) {
    case CommandKind::Play:
      if (auto r = counted(1); !r) return std::unexpected(r.error());
      return PlayArgs{.file = path_of(args[0]), .profile = options.profile};
    case CommandKind::Export:
      if (auto r = counted(2); !r) return std::unexpected(r.error());
      return ExportArgs{.input = path_of(args[0]),
                        .output = path_of(args[1]),
                        .profile = options.profile,
                        .encoder = options.encoder,
                        .burn_subtitles = options.burn_subtitles,
                        .overwrite = options.overwrite};
    case CommandKind::Bench:
      if (auto r = counted(1); !r) return std::unexpected(r.error());
      return BenchArgs{.file = path_of(args[0]), .profile = options.profile, .dump = options.dump};
    case CommandKind::ImportSdk:
      if (auto r = counted(1); !r) return std::unexpected(r.error());
      return ImportSdkArgs{.folder = path_of(args[0])};
    case CommandKind::Engines:
      return engines_command(args);
    case CommandKind::Help: {
      if (args.empty()) return HelpArgs{};
      if (auto r = counted(1); !r) return std::unexpected(r.error());
      const std::optional<CommandKind> topic = find_command(args[0]);
      if (!topic) return unknown_command(args[0]);
      return HelpArgs{.topic = topic};
    }
    case CommandKind::Version:
      if (auto r = counted(0); !r) return std::unexpected(r.error());
      return VersionArgs{};
  }
  return HelpArgs{};
}

std::expected<CommandOptions, Refusal> read_options(CommandKind kind, std::span<const Option> options) {
  CommandOptions values;
  for (const Option& option : options) {
    const OptionSpec& spec = *option.spec;
    if (!spec.applies_to(kind)) {
      return refuse(RefusalCode::OptionUnknown,
                    std::format("--{} ne s'applique pas à « {} »", spec.name, cli::spec_of(kind).name));
    }
    if (spec.value.empty() && option.has_value) {
      return refuse(RefusalCode::FieldType, std::format("--{} ne prend pas de valeur", spec.name));
    }
    if (!spec.value.empty() && option.value.empty())
      return refuse(RefusalCode::FieldMissing, cli::usage_of(spec));
    if (auto r = spec.apply(values, option.value); !r) return std::unexpected(r.error());
  }
  return values;
}

}  // namespace

namespace cli {

std::span<const CommandSpec> command_specs() {
  return kCommands;
}

std::span<const OptionSpec> option_specs() {
  return kOptions;
}

const CommandSpec& spec_of(CommandKind kind) {
  return kCommands[static_cast<std::size_t>(kind)];
}

std::string usage_of(const OptionSpec& option) {
  return option.value.empty() ? std::format("--{}", option.name)
                              : std::format("--{}={}", option.name, option.value);
}

}  // namespace cli

std::expected<Invocation, Refusal> parse_invocation(std::span<const std::string_view> args) {
  auto scan = scan_arguments(args);
  if (!scan) return std::unexpected(scan.error());
  auto debug = core::parse_command_line(scan->debug);
  if (!debug) return std::unexpected(debug.error());

  Invocation invocation{.debug = debug->debug, .data_dir = scan->data_dir};
  std::span<const std::string_view> rest = scan->positionals;
  std::optional<CommandKind> kind;
  if (!rest.empty()) {
    kind = find_command(rest.front());
    if (kind) {
      rest = rest.subspan(1);
    } else if (looks_like_path(rest.front())) {
      kind = CommandKind::Play;  // « ajp episode01.mkv »
    } else {
      return unknown_command(rest.front());
    }
  }
  if (scan->version) {
    invocation.command = VersionArgs{};
  } else if (scan->help || !kind) {
    if (!kind && !scan->options.empty()) {
      return refuse(RefusalCode::OptionUnknown,
                    std::format("--{} s'emploie avec une commande", scan->options.front().spec->name));
    }
    invocation.command = HelpArgs{.topic = kind};
  } else {
    auto options = read_options(*kind, scan->options);
    if (!options) return std::unexpected(options.error());
    auto command = build_command(*kind, rest, *options);
    if (!command) return std::unexpected(command.error());
    invocation.command = std::move(*command);
  }
  return invocation;
}

std::expected<Invocation, Refusal> parse_invocation(int argc, const char* const* argv) {
  std::vector<std::string_view> args;
  for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
  return parse_invocation(args);
}

CommandKind kind_of(const CliCommand& command) {
  // L'ordre des alternatives de CliCommand suit celui de CommandKind.
  return static_cast<CommandKind>(command.index());
}

std::string_view command_name(CommandKind kind) {
  return cli::spec_of(kind).name;
}

std::string_view encoder_name(ExportEncoder encoder) {
  for (const auto& [name, value] : kEncoders) {
    if (value == encoder) return name;
  }
  return "";
}

}  // namespace ajp::app
