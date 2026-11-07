// Outils du système pour installer moteurs et SDK : curl et tar à des emplacements connus (le
// lanceur ne cherche pas dans PATH), la progression de curl, une commande attendue.
#pragma once

#include <expected>
#include <filesystem>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"
#include "prepare/process.h"

namespace ajp::app::play {

std::optional<std::filesystem::path> system_tool(std::string_view name);

// La dernière fraction « 43.2% » d'une ligne de la barre de progression de curl.
std::optional<float> curl_percent(std::string_view line);

// P3.7 : les options de curl d'un gros téléchargement : cinq essais sur toute erreur, abandon d'une connexion
// sous 1 Ko/s pendant une minute (elle se reprend à l'essai suivant), et le proxy du système sous Windows
// (curl.exe ne lit pas les réglages de WinINET ; ailleurs, curl lit https_proxy).
std::vector<std::string> curl_download_args();
// Le proxy manuel de Windows (« hôte:port », ou la valeur https= d'une liste par protocole) ; rien sans proxy
// manuel ou hors Windows. Exposé pour les tests.
std::optional<std::string> proxy_from_setting(std::string_view setting);

// Lance spec et attend sa fin ; refus Cancelled, DecodeFailed (code non nul : what, la fin et
// la dernière ligne), ceux du lanceur.
std::expected<void, Refusal> run_tool(const prepare::ProcessSpec& spec, const prepare::LineSink& on_line,
                                      const std::stop_token& stop, std::string_view what);

}  // namespace ajp::app::play
