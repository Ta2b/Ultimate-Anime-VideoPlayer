// Le fil des mises à jour du lecteur (updater.h) : la vérification en arrière-plan, hors lecture.
#pragma once

#include <atomic>
#include <filesystem>
#include <thread>

namespace ajp::app::play {

struct PlayerPaths;

// Au lancement si elle est due, puis relue toutes les 30 minutes tant que le lecteur reste ouvert (la
// fréquence « au lancement » ne revérifie pas) ; pendant une lecture (playing), elle attend la fin de la
// vidéo. Le mode et la fréquence se relisent dans play.toml à chaque tour (A18-03) : « Jamais » choisi dans
// l'assistant ou dans Général vaut tout de suite. Le fil s'arrête avec le lecteur.
std::jthread start_update_check(const std::filesystem::path& settings_path, const PlayerPaths& paths,
                                std::atomic<bool>& playing);

}  // namespace ajp::app::play
