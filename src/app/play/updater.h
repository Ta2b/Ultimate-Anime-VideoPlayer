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
std::string_view installed_version();
// Le fichier lu d'un coup (vide s'il manque), et l'heure Unix : partagés par updater.cpp et
// updater_state.cpp.
std::string read_all(const std::filesystem::path& file);
std::int64_t unix_now();

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

// Un téléchargement : l'adresse dans le fichier, en max_seconds au plus. Celui du lecteur est curl ; un test
// donne le sien.
using UpdateFetch = std::function<std::expected<void, Refusal>(
    std::string_view url, const std::filesystem::path& file, int max_seconds, const std::stop_token& stop)>;

std::string format_update_state(const UpdateState& state);
std::expected<UpdateState, Refusal> parse_update_state(std::string_view toml_text);
// Oublie l'annonce d'une version déjà installée (ou plus ancienne, ou illisible) : après l'installation,
// la notice ne revient pas jusqu'à la vérification suivante, qui peut tarder d'une semaine (B26-01).
void forget_installed_announce(UpdateState& state, std::string_view installed);
// Absent : l'état vide. Illisible : l'état vide, mais avec les clés révoquées qu'on y lit encore (B26-10) ;
// save_update_state garde alors une copie du fichier dans <fichier>.bak. L'annonce d'une version
// qui n'est plus nouvelle est oubliée.
UpdateState load_update_state(const std::filesystem::path& file);
// Comme load_update_state, mais un fichier présent qui ne se lit pas (tenu, droits) est un refus, pas l'état
// vide (C36-03).
std::expected<UpdateState, Refusal> read_update_state(const std::filesystem::path& file);
// Réécrit le fichier : les champs de state, plus les clés inconnues d'une version plus récente (A36-05).
std::expected<void, Refusal> save_update_state(const std::filesystem::path& file, const UpdateState& state);
// Le texte à écrire : state, puis les clés de previous que cette version ne connaît pas (hors les champs des
// versions qui s'appliquaient elles-mêmes).
std::string merge_update_state(std::string_view previous, const UpdateState& state);
// Lire, changer, réécrire si quelque chose a changé, sans qu'un autre fil du processus s'intercale entre la
// lecture et l'écriture (A26-10). change est court : jamais de réseau sous ce verrou.
std::expected<void, Refusal> change_update_state(const std::filesystem::path& file,
                                                 const std::function<void(UpdateState&)>& change);
// Le fil des mises à jour : une vérification si may_check (premier tour, ou rien ne joue) et si elle est due,
// gardée dans file et journalisée.
void check_updates_if_due(const std::filesystem::path& file, UpdateMode mode, UpdateFrequency frequency,
                          bool may_check, std::int64_t now_unix, const std::stop_token& stop);
// « Check now » : une vérification tout de suite, gardée dans file ; le refus d'écriture compte après
// une vérification réussie.
std::expected<std::string, Refusal> check_updates_now(const std::filesystem::path& file,
                                                      const std::stop_token& stop);
// Les mêmes, avec ce téléchargement (essais). Le verrou de l'état n'est tenu que pour lire puis réécrire le
// fichier, jamais pendant le téléchargement (B26-09, B30-02) : ce qu'une autre tâche y change entre-temps
// reste, la vérification n'en reporte que ses propres champs.
void check_updates_if_due(const std::filesystem::path& file, UpdateMode mode, UpdateFrequency frequency,
                          bool may_check, std::int64_t now_unix, const std::stop_token& stop,
                          const UpdateFetch& fetch);
std::expected<std::string, Refusal> check_updates_now(const std::filesystem::path& file,
                                                      const std::stop_token& stop, const UpdateFetch& fetch);

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

// Où mène « Download page » : la page de notes de la release si elle est sous https://github.com/<repo>/,
// sinon la page des releases du dépôt (un manifeste, ou update.toml, ne mène pas l'utilisateur ailleurs).
std::string download_page_url(std::string_view notes_url, std::string_view repo);

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
std::expected<std::string, Refusal> run_update_check(UpdateState& state, const std::stop_token& stop,
                                                     const UpdateFetch& fetch, std::int64_t now_unix);

}  // namespace ajp::app::play
