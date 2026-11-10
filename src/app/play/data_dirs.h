#pragma once

#include <filesystem>
#include <functional>
#include <span>
#include <string>

namespace ajp::app::play {

// Le dossier des données lourdes (P3.15) : moteurs construits, paquets, modèles, téléchargements, caches.
// Sous Windows, %LOCALAPPDATA%\ajp\ajp (FOLDERID_LocalAppData), pour que plusieurs Go ne suivent pas un
// profil itinérant ; les réglages restent dans le dossier de SDL_GetPrefPath (Roaming). Ailleurs, le même
// dossier.
std::filesystem::path local_data(const std::filesystem::path& roaming);

// Déplace vers `to` les données lourdes laissées dans `from` par une version d'avant, une à une, quand `to`
// ne les a pas déjà ; `said` reçoit une ligne par entrée déplacée ou refusée. Rend le nombre d'entrées
// déplacées. D'un volume à l'autre, la copie se fait dans « <nom>.migration », renommé une fois entier ; un
// « .migration » laissé par une copie coupée est retiré et la copie recommence (A19-01).
int move_heavy_data(const std::filesystem::path& from, const std::filesystem::path& to,
                    const std::function<void(const std::string&)>& said = {});

// Le dossier des caches qui se refont seuls (moteurs construits, modèles dérivés, plans, cache des shaders).
// Sous Linux, <home>/.cache/ajp : les outils de sauvegarde l'excluent, pas ~/.local/share (A36-08) ;
// $XDG_CACHE_HOME n'est pas lu (aucune variable d'environnement). Ailleurs, et sans dossier personnel :
// heavy.
std::filesystem::path cache_data(const std::filesystem::path& home, const std::filesystem::path& heavy);
// Comme move_heavy_data, pour les seules entrées de cache : de heavy vers le dossier de cache_data.
int move_cache_data(const std::filesystem::path& from, const std::filesystem::path& to,
                    const std::function<void(const std::string&)>& said = {});

// P3.70 : les fichiers qu'une version d'avant écrivait par erreur dans le dossier parent des données
// (%APPDATA% sous Windows, ~/.local/share ailleurs), un dossier partagé avec d'autres programmes. Seuls ceux
// que le lecteur reconnaît à leur contenu partent : update.toml (sa clé last_check_unix), updates/ (rien que
// des versions, extract et previous), plans/*.shots, shaders.cache (l'en-tête du cache de libplacebo) et
// engines-relaunched. Rend le nombre d'entrées retirées.
int purge_misplaced(const std::filesystem::path& shared,
                    const std::function<void(const std::string&)>& said = {});

// Le dossier <données>/updates des versions qui se mettaient à jour elles-mêmes (archives préparées, extract,
// copie previous de l'ancienne installation) : la mise à jour n'est plus qu'une notification, il se retire au
// lancement. Rien ne part s'il contient autre chose que des versions, extract et previous. Rend 1 s'il est
// retiré.
int purge_update_leftovers(const std::filesystem::path& data,
                           const std::function<void(const std::string&)>& said = {});

}  // namespace ajp::app::play
