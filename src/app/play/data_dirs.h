#pragma once

#include <filesystem>
#include <functional>
#include <string>

namespace ajp::app::play {

// Le dossier des données lourdes (P3.15) : moteurs construits, paquets, modèles, téléchargements, caches.
// Sous Windows, %LOCALAPPDATA%\ajp\ajp (FOLDERID_LocalAppData), pour que plusieurs Go ne suivent pas un
// profil itinérant ; les réglages restent dans le dossier de SDL_GetPrefPath (Roaming). Ailleurs, le même
// dossier.
std::filesystem::path local_data(const std::filesystem::path& roaming);

// Déplace vers `to` les données lourdes laissées dans `from` par une version d'avant, une à une, quand `to`
// ne les a pas déjà ; `said` reçoit une ligne par entrée déplacée ou refusée. Rend le nombre d'entrées
// déplacées.
int move_heavy_data(const std::filesystem::path& from, const std::filesystem::path& to,
                    const std::function<void(const std::string&)>& said = {});

}  // namespace ajp::app::play
