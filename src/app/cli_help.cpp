// Textes d'aide, tirés des mêmes tables que l'analyse (cli_spec.h).
#include <algorithm>
#include <format>
#include <string>

#include "app/cli.h"
#include "app/cli_spec.h"
#include "core/debug_flags.h"

namespace ajp::app {
namespace {

// Section « Options de débogage » de core, sans sa ligne d'usage ni son --help,
// que l'aide générale porte déjà.
std::string debug_section() {
  std::string text = core::command_line_help();
  const std::size_t start = text.find("\n\n");
  if (start != std::string::npos) text.erase(0, start + 2);
  const std::size_t help = text.find("\n  --help ");
  if (help != std::string::npos) text.erase(help + 1);
  return text;
}

// Lignes « gauche  droite » alignées sur la plus longue colonne de gauche.
std::string columns(const std::vector<std::pair<std::string, std::string>>& rows) {
  std::size_t width = 0;
  for (const auto& row : rows) width = std::max(width, row.first.size());
  std::string text;
  for (const auto& [left, right] : rows) text += std::format("  {:<{}}  {}\n", left, width, right);
  return text;
}

std::string options_of(CommandKind kind) {
  std::vector<std::pair<std::string, std::string>> rows;
  for (const cli::OptionSpec& option : cli::option_specs()) {
    if (option.applies_to(kind)) rows.emplace_back(cli::usage_of(option), std::string(option.help));
  }
  return rows.empty() ? std::string{} : "\nOptions :\n" + columns(rows);
}

}  // namespace

std::string help_text(std::string_view default_data_dir) {
  std::string text =
      "ajp : lecteur vidéo anime, avec upscaling, interpolation et HDR en temps réel.\n\n"
      "Usage :\n"
      "  ajp [options] <fichier>               lit une vidéo (comme « ajp play <fichier> »)\n"
      "  ajp [options] <commande> [arguments]\n\n"
      "Commandes :\n";
  std::vector<std::pair<std::string, std::string>> commands;
  for (const cli::CommandSpec& spec : cli::command_specs()) {
    commands.emplace_back(std::string(spec.usage), std::string(spec.summary));
  }
  text += columns(commands);

  text += "\nOptions générales :\n";
  text += columns({
      {"--data-dir=DOSSIER",
       std::format("dossier des réglages, modèles et moteurs (défaut : {})", default_data_dir)},
      {"--help, -h", "cette aide ; « ajp <commande> --help » pour une commande"},
      {"--version", "version du lecteur"},
  });

  std::vector<std::pair<std::string, std::string>> command_options;
  for (const cli::OptionSpec& option : cli::option_specs()) {
    std::string users;
    for (const cli::CommandSpec& spec : cli::command_specs()) {
      if (option.applies_to(spec.kind)) users += std::format("{}{}", users.empty() ? "" : ", ", spec.name);
    }
    command_options.emplace_back(cli::usage_of(option), std::format("{} ({})", option.help, users));
  }
  text += "\nOptions des commandes :\n" + columns(command_options);
  text += "\n" + debug_section();
  text +=
      "\nExemples :\n"
      "  ajp episode01.mkv\n"
      "  ajp export episode01.mkv episode01-4k.mkv --profile=2\n"
      "  ajp engines install\n";
  return text;
}

std::string help_text(CommandKind kind) {
  const cli::CommandSpec& spec = cli::spec_of(kind);
  return std::format("Usage : ajp {}\n\n{}\n", spec.usage, spec.details) + options_of(kind) +
         std::format("\nExemple :\n  {}\n\nOptions générales et de débogage : ajp --help\n", spec.example);
}

}  // namespace ajp::app
