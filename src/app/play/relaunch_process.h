// Le lancement du lecteur relancé (main.cpp : relance après la mise à jour d'un moteur ou la perte du GPU).
#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace ajp::app::play {

// Lance `executable` avec `arguments` (sans argv[0]). Rien d'ouvert ne passe au lecteur relancé (A18-12) : un
// contexte CUDA gardé par un descripteur hérité retiendrait sa mémoire vidéo.
//   - replace : sous POSIX, execv remplace ce processus et ne revient que sur un échec ; sous Windows, un
//     nouveau processus démarre (_spawnv) et la main revient, celui-ci n'a plus qu'à sortir ;
//   - !replace (C34-04, relaunch_target.h : exit_stuck) : un nouveau processus sous POSIX aussi
//     (posix_spawn), puis la main revient. execve dans un processus à plusieurs threads attend la fin de tous
//     les autres : un thread bloqué dans le pilote du GPU (sommeil non interruptible) bloquait la relance et
//     la sortie qui la suit, quand Windows relançait.
// Rend vide sur un lancement réussi, sinon la cause, pour le journal.
std::string start_relaunched(const std::filesystem::path& executable,
                             const std::vector<std::string>& arguments, bool replace);

// Un argument cité pour la ligne de commande de Windows, à la façon de CommandLineToArgvW : _spawnv ne met
// aucun argument entre guillemets, un chemin à espaces (« C:\Mes vidéos\a.mkv ») arriverait coupé en deux.
// Pur.
std::string windows_quoted(const std::string& argument);

}  // namespace ajp::app::play
