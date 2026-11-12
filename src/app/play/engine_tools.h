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
// -q en tête (aucun .curlrc), https seulement, redirections comprises (A35-05). file:// n'est permis que
// pour les tests (`allow_local_files`) : l'index exige déjà https, et un lien file:// qui s'y glisserait
// ne doit pas lire un fichier de la machine (B06-10, B35-09). Puis curl_system_args().
std::vector<std::string> curl_download_args(bool allow_local_files = false, std::string_view url = {});
// Ce que curl demande pour passer comme le système sous Windows : le proxy manuel ou celui d'un script PAC
// (détection automatique comprise, B06-08), et la révocation du
// certificat vérifiée au mieux (--ssl-revoke-best-effort) : Schannel refuse sinon toute connexion dont la
// révocation ne se vérifie pas (antivirus qui inspecte le HTTPS, OCSP bloqué ; A06-13). Rien ailleurs.
std::vector<std::string> curl_system_args(std::string_view url = {});
// L'hôte d'une URL (« github.com » pour « https://github.com/x »), vide sans « :// » : le proxy se demande
// pour lui.
std::string url_host(std::string_view url);
// Le proxy manuel de Windows (« hôte:port », ou la valeur https= d'une liste par protocole) ; rien sans proxy
// manuel ou hors Windows. Exposé pour les tests.
std::optional<std::string> proxy_from_setting(std::string_view setting);

// Lance spec et attend sa fin ; refus Cancelled, DecodeFailed (code non nul : what, la fin et
// la dernière ligne), ceux du lanceur.
std::expected<void, Refusal> run_tool(const prepare::ProcessSpec& spec, const prepare::LineSink& on_line,
                                      const std::stop_token& stop, std::string_view what);

}  // namespace ajp::app::play
