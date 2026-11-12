// La vidéo à rouvrir après une relance du lecteur (A31-02, A29-04) : une relance rejoue les arguments du
// lancement (main.cpp), qui ne nomment pas la vidéo en cours. Le lecteur qui se relance écrit ici la vidéo,
// l'instant et la pause, et pourquoi ; le suivant les lit une fois, puis efface le fichier.
//
// Un fichier texte (relaunch.txt, dans le dossier des données), une clé par ligne : « file », « start_us »,
// « paused », « reason » (« gpu-lost », « engine-updated ») et « written_s » (secondes depuis l'époque).
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "core/types.h"

namespace ajp::app::play {

enum class RelaunchReason : std::uint8_t { GpuLost, EngineUpdated };

struct RelaunchTarget {
  std::filesystem::path file;
  // C18-02 : absent quand la vidéo n'avait pas encore d'instant imposé (relance à l'ouverture) : la reprise
  // de lecture retenue s'applique alors comme à une ouverture ordinaire.
  std::optional<std::int64_t> start_us;
  bool paused = false;
  RelaunchReason reason = RelaunchReason::EngineUpdated;
  std::int64_t written_s = 0;  // quand la relance a été demandée
};

// Plus vieux que ceci, le fichier est d'une autre séance (relance qui n'a pas abouti) : ignoré.
inline constexpr std::int64_t kRelaunchFreshS = 120;
// Une perte du GPU dans la minute qui suit une relance pour la même raison ne relance plus (A31-02) : le
// message et l'accueil.
inline constexpr std::int64_t kRelaunchAgainS = 60;

std::filesystem::path relaunch_target_file(const std::filesystem::path& data);
std::int64_t relaunch_now_s();

std::expected<void, Refusal> write_relaunch_target(const std::filesystem::path& store,
                                                   const RelaunchTarget& target);
// Lu puis effacé ; rien si le fichier manque, est illisible ou plus vieux que kRelaunchFreshS à now_s.
std::optional<RelaunchTarget> take_relaunch_target(const std::filesystem::path& store, std::int64_t now_s);

// La relance demandée (*paths.relaunch) avec la vidéo à rouvrir : vrai si elle l'est. Sans file, la relance
// rejoue les arguments du lancement comme avant.
struct PlayerPaths;
bool ask_relaunch(const PlayerPaths& paths, const std::filesystem::path& file,
                  std::optional<std::int64_t> start_us, bool paused, RelaunchReason reason);

// La vidéo du lecteur qui s'est relancé : options.file devient elle. Rien pour un essai ou un banc. Seul le
// processus relancé (--relaunched=) prend le fichier : un lancement de l'utilisateur, même dans les deux
// minutes d'une relance ratée, ouvre sa vidéo à lui (B18-02, B36-10) et retire le fichier resté.
struct PlayOptions;
std::optional<RelaunchTarget> take_relaunch(PlayOptions& options, const PlayerPaths& paths,
                                            std::int64_t now_s);

// B31-04 : le GPU ne répond plus pendant une fermeture (démontage de la chaîne ou de l'écran) que rien ne
// borne. Le garde (close_watch.h) sort alors du processus ; une relance demandée par ask_relaunch (perte du
// GPU) n'est pas perdue : la fonction posée par main relance le lecteur avant la sortie. Sans elle, la
// sortie seule (75, EX_TEMPFAIL : la machine, pas la vidéo), et relaunch.txt expirait sans personne pour le
// lire.
void set_stuck_relaunch(void (*relaunch)());
[[noreturn]] void exit_stuck(std::string_view why);
// C30-05 : le démontage de la chaîne bloqué ; la cause dite au journal n'est pas seulement le GPU.
[[noreturn]] void exit_chain_stuck();
inline constexpr int kStuckExitCode = 75;

}  // namespace ajp::app::play
