// Ce dont la relance du lecteur a besoin pour lancer le processus suivant (main.cpp) : les arguments rejoués,
// leur citation pour _spawnv sous Windows, et les descripteurs fermés au passage sous POSIX. Hors de main.cpp
// pour être testés (C18-10).
#pragma once

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "app/play/options.h"

namespace ajp::app::play {

// Les arguments du lancement, sans un --relaunched= d'une relance d'avant, puis celui de cette relance : le
// lecteur relancé sait qu'il l'est et pourquoi (options.h).
std::vector<std::string> relaunch_arguments(std::span<const std::string_view> launched, RelaunchedFor reason);

// Un argument cité à la façon de CommandLineToArgvW : _spawnv ne met aucun argument entre guillemets, et un
// chemin à espaces (« C:\Mes vidéos\a.mkv ») arriverait coupé en deux. Antislashs doublés seulement devant un
// guillemet, cité ou fermant.
std::string windows_quoted(const std::string& argument);

// POSIX : tous les descripteurs au-delà de 2 marqués FD_CLOEXEC (A18-12) ; rien d'ouvert ne passe au lecteur
// relancé (pilote NVIDIA, socket, fichiers) : un contexte CUDA gardé par un descripteur hérité retiendrait sa
// mémoire vidéo. close_range quand le noyau l'a (C18-12), sinon un balayage borné. Rien sous Windows.
void close_on_exec_all();

}  // namespace ajp::app::play
