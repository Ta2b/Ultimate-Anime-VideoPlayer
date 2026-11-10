// La notification des nouvelles versions du lecteur : une simple notification, l'utilisateur installe
// lui-même (chantier C01).
//
//   Vérification : manifest.toml et manifest.toml.sig de la dernière release (https, curl du système), la
//   signature Ed25519 vérifiée avec les clés publiques compilées dans le lecteur. Seule donnée envoyée : la
//   requête HTTPS. Une version plus récente se signale (notice et page Général > Application) avec le lien de
//   sa release ; le lecteur ne télécharge ni n'installe rien lui-même.
// Sans clé de signature compilée, rien n'est cru : la vérification dit pourquoi et ne fait rien d'autre.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

#include "app/play/update_choice.h"
#include "core/types.h"
#include "core/update_manifest.h"

namespace ajp::app::play {

// Le dépôt des releases et la clé publique (hexadécimale, 64 chiffres), posés à la compilation
// (AJP_RELEASE_REPO, AJP_UPDATE_PUBKEY) ; la clé est vide tant que la clé de signature n'existe pas.
std::string_view release_repo();
std::string_view update_public_key();
std::string_view installed_version();

// Où chercher le manifeste : celui de la dernière release (« latest »), seul canal publié.
std::string manifest_url(std::string_view repo);
// La page des releases, où l'utilisateur prend la nouvelle version.
std::string releases_page(std::string_view repo);

// Ce que le lecteur retient entre deux lancements (<données>/update.toml). Les champs des versions qui
// s'appliquaient elles-mêmes (staged_*, previous_version, unconfirmed_*…) sont ignorés à la lecture.
struct UpdateState {
  std::int64_t last_check_unix = 0;    // la dernière vérification qui a eu une réponse du serveur
  std::int64_t last_attempt_unix = 0;  // la dernière tentative, réponse ou non (nouvel essai après une panne)
  std::string latest_version;          // la plus récente vue et plus récente que l'installée ; vide : à jour
  std::string notes_url;
  std::string last_error;
  std::vector<std::string> revoked_keys;  // clés publiques qu'un manifeste vérifié a révoquées
  friend bool operator==(const UpdateState&, const UpdateState&) = default;
};

std::string format_update_state(const UpdateState& state);
std::expected<UpdateState, Refusal> parse_update_state(std::string_view toml_text);
// Absent : l'état vide.
UpdateState load_update_state(const std::filesystem::path& file);
std::expected<void, Refusal> save_update_state(const std::filesystem::path& file, const UpdateState& state);

// Après une tentative sans réponse du serveur (pas de réseau), la suivante attend au moins ce délai, pas
// toute la période choisie (A26-07).
inline constexpr std::int64_t kUpdateRetrySeconds = 3600;

// Faut-il vérifier maintenant ? frequency : « launch » (à chaque lancement), « daily », « weekly » ; mode «
// never » : jamais.
bool update_due(const UpdateState& state, UpdateMode mode, UpdateFrequency frequency, std::int64_t now_unix);

struct CheckResult {
  std::optional<core::UpdateManifest> newer;  // plus récent que le lecteur
  std::string message;                        // « à jour », « 0.7.0 disponible »…
  std::vector<std::string> revoked_keys;      // révoquées par le manifeste vérifié, à retenir
};

// Le statut HTTP que curl écrit par --write-out kCurlStatusFormat (« ajp-http 404 »), jamais lu dans ses
// messages d'erreur (P2.11) ; absent pour une autre ligne.
inline constexpr std::string_view kCurlStatusFormat = "ajp-http %{http_code}\n";
std::optional<int> curl_status_line(std::string_view line);

// Le refus lisible d'un échec de curl (code de sortie, statut HTTP, 0 sans réponse) : « pas de connexion
// Internet »… ; un fichier absent du serveur (HTTP 404) est refusé TrackUnavailable.
Refusal curl_failure(int code, int http_status);

// Télécharge et vérifie le manifeste. Refus : clé absente, réseau, manifeste absent (la dernière release n'en
// a pas : jamais lu comme « à jour », SEC-05), signature fausse, manifeste illisible ou expiré.
std::expected<CheckResult, Refusal> check_for_update(const std::stop_token& stop);

// Un téléchargement : l'adresse dans le fichier, en max_seconds au plus. Celui du lecteur est curl ; un test
// donne le sien.
using UpdateFetch = std::function<std::expected<void, Refusal>(
    std::string_view url, const std::filesystem::path& file, int max_seconds, const std::stop_token& stop)>;
// Le téléchargement du lecteur : curl du système, https seulement (un 404 est refusé TrackUnavailable).
std::expected<void, Refusal> fetch_https(std::string_view url, const std::filesystem::path& file,
                                         int max_seconds, const std::stop_token& stop);
// Les clés publiques que le lecteur croit : l'active, puis celle de secours si elle est compilée, moins les
// révoquées.
std::vector<std::string> update_public_keys(const std::vector<std::string>& revoked);
// La même vérification, avec ce téléchargement, ces clés publiques (hexadécimal) et cette version installée :
// le manifeste doit être signé par l'une d'elles, et ne vaut plus si sa propre clé y est révoquée.
std::expected<CheckResult, Refusal> check_for_update(const std::stop_token& stop, const UpdateFetch& fetch,
                                                     const std::vector<std::string>& keys,
                                                     std::string_view installed);

// La vérification en arrière-plan : met à jour state (tentative, dernière réponse, version vue, clés
// révoquées). Rend le message. À lancer hors de la lecture.
std::expected<std::string, Refusal> run_update_check(UpdateState& state, const std::stop_token& stop);
std::expected<std::string, Refusal> run_update_check(UpdateState& state, const std::stop_token& stop,
                                                     const UpdateFetch& fetch, std::int64_t now_unix);

}  // namespace ajp::app::play
