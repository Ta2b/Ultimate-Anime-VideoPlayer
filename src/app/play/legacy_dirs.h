// Les dossiers d'avant Kirari Player. Le lecteur s'appelait ajp et rangeait ses données dans deux niveaux
// (<données>/ajp/ajp/, SDL_GetPrefPath("ajp", "ajp")) ; Kirari n'en a qu'un (<données>/kirari/,
// SDL_GetPrefPath("", "kirari")). Au premier lancement, l'ancien dossier est repris tel quel : réglages,
// moteurs TensorRT déjà construits, modèles et journaux, sans rien reconstruire.
#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace kirari::app::play {

// Le nom du lecteur d'avant, pour ses dossiers seulement.
inline constexpr std::string_view kLegacyName = "ajp";
// Le nom des dossiers du lecteur : un seul niveau, sous le dossier des données du système.
inline constexpr std::string_view kDataName = "kirari";

// <base>/ajp/ajp d'après <base>/kirari (barre finale de SDL acceptée) ; vide pour un chemin vide.
std::filesystem::path legacy_pref_dir(std::filesystem::path pref);

// Déplace `legacy` à la place de `target` quand `target` est absent ou vide (SDL vient de le créer), puis
// retire le parent de `legacy` s'il est resté vide (<base>/ajp). Rien ne bouge si `legacy` manque ou si
// `target` a déjà du contenu : une reprise faite, ou un dossier Kirari neuf déjà servi, ne s'écrase pas. Un
// renommage refusé (autre volume, droits) laisse les deux dossiers tels quels. `said` reçoit une ligne pour
// le journal. Rend vrai si le dossier a été repris.
bool adopt_legacy_dir(const std::filesystem::path& legacy, const std::filesystem::path& target,
                      const std::function<void(const std::string&)>& said = {});

// Le dossier des réglages : <données>/kirari/ (SDL_GetPrefPath("", "kirari")), un seul niveau ; celui d'ajp
// y est repris au premier appel. Vide si SDL n'en donne pas.
std::filesystem::path prefs_dir();
// Ce que la reprise a fait, à dire au journal une fois ouvert.
const std::vector<std::string>& adopted_dir_lines();

}  // namespace kirari::app::play
