// Tables de la ligne de commande, partagées par l'analyse (cli.cpp) et l'aide
// (cli_help.cpp) : une commande ou une option ajoutée ici apparaît dans les deux.
// En-tête interne au module app.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string_view>

#include "app/cli.h"
#include "core/types.h"

namespace ajp::app::cli {

struct CommandSpec {
  CommandKind kind;
  std::string_view name;     // mot tapé : « import-sdk »
  std::string_view usage;    // « import-sdk <dossier> »
  std::string_view summary;  // une ligne, dans l'aide générale
  std::string_view details;  // paragraphe de l'aide de la commande
  std::string_view example;  // ligne complète, « ajp » compris
};

// Valeurs des options propres aux commandes, avant leur répartition dans *Args.
struct CommandOptions {
  std::optional<int> profile;
  std::optional<ExportEncoder> encoder;
  std::optional<std::filesystem::path> dump;
  bool burn_subtitles = false;
  bool overwrite = false;
};

using Apply = std::expected<void, Refusal> (*)(CommandOptions& options, std::string_view value);

struct OptionSpec {
  std::string_view name;   // sans « -- »
  std::string_view value;  // forme de la valeur ; vide pour un simple drapeau
  std::string_view help;
  std::uint8_t commands;  // un bit par CommandKind qui l'accepte
  Apply apply;

  constexpr bool applies_to(CommandKind kind) const {
    return (commands & (1U << static_cast<unsigned>(kind))) != 0;
  }
};

constexpr std::uint8_t bit(CommandKind kind) {
  return static_cast<std::uint8_t>(1U << static_cast<unsigned>(kind));
}

std::span<const CommandSpec> command_specs();
std::span<const OptionSpec> option_specs();
const CommandSpec& spec_of(CommandKind kind);

// « --profile=1..9 » ou « --overwrite ».
std::string usage_of(const OptionSpec& option);

}  // namespace ajp::app::cli
