// Ligne de commande d'ajp : une commande, ses arguments et ses options, plus les
// options de débogage de core::DebugFlags. Tout vient d'argv, rien de
// l'environnement. Analyse pure : aucun fichier n'est ouvert ni vérifié ici.
//
//   ajp [options] <fichier>                  lit une vidéo (raccourci de « play »)
//   ajp [options] play <fichier>
//   ajp [options] export <fichier> <sortie>
//   ajp [options] bench <fichier>
//   ajp [options] import-sdk <dossier>
//   ajp [options] engines [install [moteur…]]
//   ajp help [commande]   ajp --help   ajp --version
//
// Les options générales et de débogage se placent n'importe où ; celles d'une
// commande seulement pour elle. « -- » termine les options.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "core/debug_flags.h"
#include "core/engines.h"
#include "core/types.h"

namespace ajp::app {

enum class CommandKind : std::uint8_t { Play, Export, Bench, ImportSdk, Engines, Help, Version };

// Encodeurs du plan (section export) ; export les branchera un par un.
enum class ExportEncoder : std::uint8_t { NvencHevc, NvencAv1, SvtAv1, VideoToolboxHevc };

struct PlayArgs {
  std::filesystem::path file;
  std::optional<int> profile;  // --profile=1..9 : remplace le profil de démarrage
};

struct ExportArgs {
  std::filesystem::path input;
  std::filesystem::path output;
  std::optional<int> profile;
  std::optional<ExportEncoder> encoder;  // absent : celui du GPU détecté
  bool burn_subtitles = false;           // grave les sous-titres au lieu de recopier leur piste
  bool overwrite = false;                // remplace une sortie existante
};

struct BenchArgs {
  std::filesystem::path file;
  std::optional<int> profile;
  std::optional<std::filesystem::path> dump;  // --dump=FICHIER : framemd5 des slots du ring
};

struct ImportSdkArgs {
  std::filesystem::path folder;
};

struct EnginesArgs {
  bool install = false;                 // « engines install »
  std::vector<core::EngineId> engines;  // nommés ; vide : la présélection
};

struct HelpArgs {
  std::optional<CommandKind> topic;  // absent : l'aide générale
};

struct VersionArgs {};

using CliCommand =
    std::variant<PlayArgs, ExportArgs, BenchArgs, ImportSdkArgs, EnginesArgs, HelpArgs, VersionArgs>;

struct Invocation {
  CliCommand command = HelpArgs{};
  core::DebugFlags debug;
  std::optional<std::filesystem::path> data_dir;  // --data-dir : remplace le dossier donné par SDL
};

// Arguments sans argv[0]. Commande, argument ou option inconnus, valeur hors
// plage : refus nommé, dont le détail dit quoi taper.
std::expected<Invocation, Refusal> parse_invocation(std::span<const std::string_view> args);
// Commodité pour main() : argv tel que le système le passe, argv[0] compris.
std::expected<Invocation, Refusal> parse_invocation(int argc, const char* const* argv);

CommandKind kind_of(const CliCommand& command);
std::string_view command_name(CommandKind kind);  // « play », « import-sdk »…
std::string_view encoder_name(ExportEncoder encoder);

// Aide générale ; default_data_dir s'affiche comme défaut de --data-dir.
std::string help_text(std::string_view default_data_dir);
// Aide d'une commande : usage, ce qu'elle fait, ses options, un exemple.
std::string help_text(CommandKind kind);

}  // namespace ajp::app
