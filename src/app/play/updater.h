// La mise à jour automatique signée du lecteur (docs/ux-parametres.md, 4.5).
//
//   1. Vérification : manifest.toml et manifest.toml.sig de la release (https, curl du système), la signature
//      Ed25519 vérifiée avec la clé publique compilée dans le lecteur. Seule donnée envoyée : la requête
//      HTTPS.
//   2. Préparation : l'archive du système, téléchargée à côté, sa taille et son SHA-256 comparés au
//   manifeste.
//   3. Application : au prochain lancement, jamais en pleine lecture ; l'archive est vérifiée de nouveau
//      (taille, SHA-256) et doit contenir le lecteur, puis l'ancienne version est copiée dans
//      <données>/updates/previous pour le retour arrière. Une mise à jour ratée laisse l'ancienne en place.
//   4. Garde : la nouvelle version n'est confirmée qu'après une lecture menée à son terme ; après
//      kMaxUnconfirmedLaunches lancements sans confirmation (elle plante au démarrage), l'ancienne revient
//      d'elle-même.
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

#include "core/types.h"
#include "core/update_manifest.h"

namespace ajp::app::play {

// Le dépôt des releases et la clé publique (hexadécimale, 64 chiffres), posés à la compilation
// (AJP_RELEASE_REPO, AJP_UPDATE_PUBKEY) ; la clé est vide tant que la clé de signature n'existe pas.
std::string_view release_repo();
std::string_view update_public_key();
std::string_view installed_version();

// Où chercher le manifeste : « stable » → la dernière release, « preview » → la release de tag « preview ».
std::string manifest_url(std::string_view repo, std::string_view channel);

// Ce que le lecteur retient entre deux lancements (<données>/update.toml).
struct UpdateState {
  std::int64_t last_check_unix = 0;
  std::string latest_version;  // la plus récente vue et plus récente que l'installée ; vide : à jour
  std::string notes_url;
  std::string staged_version;    // préparée, à appliquer au prochain lancement
  std::string staged_file;       // son archive, sous <données>/updates
  std::string previous_version;  // celle que le retour arrière rend
  std::string last_error;
  std::string staged_sha256;  // de l'archive préparée, revérifiée avant l'application
  std::uint64_t staged_size = 0;
  std::string unconfirmed_version;  // appliquée, pas encore confirmée par une lecture menée à son terme
  int unconfirmed_launches = 0;     // ses lancements depuis
  std::string skipped_version;      // retirée par la garde : signalée, plus jamais préparée d'elle-même
  friend bool operator==(const UpdateState&, const UpdateState&) = default;
};

std::string format_update_state(const UpdateState& state);
std::expected<UpdateState, Refusal> parse_update_state(std::string_view toml_text);
// Absent : l'état vide.
UpdateState load_update_state(const std::filesystem::path& file);
std::expected<void, Refusal> save_update_state(const std::filesystem::path& file, const UpdateState& state);

// Faut-il vérifier maintenant ? frequency : « launch » (à chaque lancement), « daily », « weekly » ; mode «
// never » : jamais.
bool update_due(const UpdateState& state, std::string_view mode, std::string_view frequency,
                std::int64_t now_unix);

struct UpdatePaths {
  std::filesystem::path data;     // <données>
  std::filesystem::path install;  // le dossier du lecteur (celui de l'exécutable)
  std::filesystem::path executable;
};

struct CheckResult {
  std::optional<core::UpdateManifest>
      newer;            // plus récent que le lecteur, de ce canal, avec une archive pour ce système
  std::string message;  // « à jour », « 0.7.0 disponible »…
};

// Le statut HTTP d'un message d'erreur de curl (« … returned error: 404 ») ; 0 : aucun.
int curl_http_status(std::string_view message);

// Le refus lisible d'un échec de curl (code de sortie, dernière ligne écrite) : « pas de connexion Internet
// »… ; un fichier absent du serveur (HTTP 404) est refusé TrackUnavailable.
Refusal curl_failure(int code, std::string_view message);

// Télécharge et vérifie le manifeste. Refus : clé absente, réseau, signature fausse, manifeste illisible. Un
// canal sans manifeste publié (HTTP 404) n'est pas un refus : le lecteur est à jour.
std::expected<CheckResult, Refusal> check_for_update(std::string_view channel, const std::stop_token& stop);

// Télécharge l'archive du manifeste sous <données>/updates, vérifie taille et SHA-256 ; rend son chemin.
std::expected<std::filesystem::path, Refusal> download_update(const UpdatePaths& paths,
                                                              const core::UpdateManifest& manifest,
                                                              const std::stop_token& stop);

// Applique l'archive préparée : copie l'installation actuelle dans previous, extrait l'archive sur
// l'installation. Rend le message ; en cas d'échec l'installation d'avant est remise. L'état est mis à jour
// (pas écrit).
std::expected<std::string, Refusal> apply_staged_update(const UpdatePaths& paths, UpdateState& state);

// Remet la version gardée dans previous. Même contrat.
std::expected<std::string, Refusal> roll_back_update(const UpdatePaths& paths, UpdateState& state);

// Lancements d'une version appliquée sans lecture menée à son terme, au-delà desquels elle est retirée.
inline constexpr int kMaxUnconfirmedLaunches = 3;

// Au lancement (étape 4) : compte un lancement de la version appliquée pas encore confirmée ; au-delà de
// kMaxUnconfirmedLaunches, remet l'ancienne (rend son message : le lecteur se relance). Retire aussi les
// « .old » qu'une application a laissés (Windows). L'état est mis à jour (pas écrit).
std::optional<std::string> guard_new_version(const UpdatePaths& paths, UpdateState& state,
                                             std::string_view running_version);
// Une lecture menée à son terme : la version en cours est bonne. Vrai si l'état a changé.
bool confirm_version(UpdateState& state, std::string_view running_version);

// Pourquoi ce lecteur ne se met pas à jour lui-même : « mis à jour par votre gestionnaire de paquets »
// (Flatpak, dossier en lecture seule). Vide : il le peut.
std::string update_managed_reason(const UpdatePaths& paths);

// La vérification en arrière-plan (updater.h : étapes 1 et 2) : met à jour state (dernière vérification,
// version vue, archive préparée si mode == « auto »). Rend le message. À lancer hors de la lecture.
std::expected<std::string, Refusal> run_update_check(const UpdatePaths& paths, std::string_view mode,
                                                     std::string_view channel, UpdateState& state,
                                                     const std::stop_token& stop);

// Le rythme de l'extraction : une archive tar.gz (ou zip sous Windows, par tar.exe) dont la racine est un
// dossier.
std::expected<void, Refusal> extract_archive(const std::filesystem::path& archive,
                                             const std::filesystem::path& into);

}  // namespace ajp::app::play
