// Outils du système pour installer moteurs et SDK : curl et tar à des emplacements connus (le
// lanceur ne cherche pas dans PATH), la progression de curl, une commande attendue.
#pragma once

#include <expected>
#include <filesystem>
#include <optional>
#include <stop_token>
#include <string_view>

#include "core/types.h"
#include "prepare/process.h"

namespace ajp::app::play {

std::optional<std::filesystem::path> system_tool(std::string_view name);

// La dernière fraction « 43.2% » d'une ligne de la barre de progression de curl.
std::optional<float> curl_percent(std::string_view line);

// Lance spec et attend sa fin ; refus Cancelled, DecodeFailed (code non nul : what, la fin et
// la dernière ligne), ceux du lanceur.
std::expected<void, Refusal> run_tool(const prepare::ProcessSpec& spec, const prepare::LineSink& on_line,
                                      const std::stop_token& stop, std::string_view what);

}  // namespace ajp::app::play
