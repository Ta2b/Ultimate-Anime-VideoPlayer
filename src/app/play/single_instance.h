// Un seul lecteur par dossier de réglages (P2.13). Le premier lancement tient le canal : un canal nommé
// sous Windows (FILE_FLAG_FIRST_PIPE_INSTANCE, la prise est atomique), un socket Unix ailleurs. Un
// lancement suivant lui passe sa vidéo, attend son accusé et se ferme : la vidéo s'ouvre dans la fenêtre
// déjà là, comme glissée dessus. Deux lecteurs n'écrivent plus play.toml, resume.tsv et le journal en même
// temps, ni n'appliquent la même mise à jour.
//
// Protocole : une ligne UTF-8 par fichier, puis une ligne vide ; le premier répond « ok ».
#pragma once

#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ajp::app::play {

// La source (SDL_DropEvent::source) d'un fichier venu d'un autre lancement : la fenêtre passe devant.
inline constexpr std::string_view kForwardedDrop = "ajp-instance";

// Un lancement simple, double-clic ou « Ouvrir avec » : au plus des fichiers, aucune option. Les essais du
// banc, les captures et toute option gardent leur propre processus. Un lecteur relancé (--relaunched=)
// reprend la place de celui qu'il remplace.
bool shares_instance(std::span<const std::string_view> args);

// Le nom du canal pour ce dossier de réglages : stable d'une version à l'autre (FNV-1a de son chemin).
std::string instance_key(const std::filesystem::path& settings_dir);

class InstanceListener;

struct InstanceClaim {
  // Vrai : un autre lecteur a reçu les fichiers, ce lancement s'arrête là.
  bool forwarded = false;
  // Le canal tenu par ce lancement, qui pousse les fichiers reçus dans la file de SDL ; nul s'il n'a pu être
  // pris (le lecteur démarre quand même, sans instance unique).
  std::shared_ptr<InstanceListener> listener;
};

// Prend le canal ou passe files au lecteur qui le tient. channel_dir porte le socket Unix (ignoré sous
// Windows). Ne lève rien ; un échec du système laisse le lancement démarrer seul.
InstanceClaim claim_instance(const std::filesystem::path& channel_dir, std::string_view key,
                             const std::vector<std::string>& files);

// Remet des fichiers dans la file de SDL, comme un glisser-déposer venu d'un autre lancement (la fenêtre
// passe devant). Faux si SDL en refuse un. Sert aussi à rendre à la boucle suivante les vidéos reçues
// pendant une attente qui ne les ouvre pas (DeferredDrops, B29-04).
// activation_token : le jeton d'activation Wayland du lancement qui transmet (C34-15), gardé pour
// take_forwarded_activation.
bool push_forwarded_files(const std::vector<std::string>& files, std::string_view activation_token = {});

// Le dernier jeton d'activation reçu, vidé ; vide s'il n'y en a pas.
std::string take_forwarded_activation();

// Les fichiers d'un message (lignes jusqu'à la ligne vide), puis son jeton d'activation éventuel
// (« activation=<jeton> » après la ligne vide, C34-15) ; exposé pour les tests.
std::vector<std::string> instance_message_files(std::string_view message);
std::string instance_message_token(std::string_view message);
std::string instance_message(const std::vector<std::string>& files, std::string_view activation_token = {});

}  // namespace ajp::app::play
