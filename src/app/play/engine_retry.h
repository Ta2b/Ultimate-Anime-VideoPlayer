// Un moteur choisi dont l'installation a échoué pour une panne passagère (coupure du réseau, 5xx de GitHub,
// portail captif) reste choisi : Vulkan prend le relais pour cette session, et le lancement suivant le
// réinstalle, le téléchargement repris là où il s'était arrêté (C06-02). Seul un refus définitif (contenu
// faux, place manquante, droits) le retire des moteurs choisis.
#pragma once

#include <chrono>
#include <filesystem>

#include "core/settings.h"
#include "core/types.h"

namespace ajp::app::play {

// Vrai si un nouvel essai au lancement suivant peut aboutir : tout échec sauf un contenu qui ne correspond
// pas à l'index (HashMismatch), le manque de place, des droits d'écriture refusés ou un SDK refusé, que rien
// ne change d'un lancement à l'autre et qui retélécharge des centaines de Mo pour rien.
bool install_failure_is_transient(RefusalCode code);

// engines après un échec passager : le moteur reste choisi, n'est plus installé, et les moteurs préférés
// retombent sur ce qui est installé.
core::Settings postponed_engine(core::Settings engines, core::EngineId id);

// engines après l'installation de id : installé, et préféré s'il est le plus rapide des choisis installés.
core::Settings installed_engine(core::Settings engines, core::EngineId id);

// Les moteurs choisis et non installés, que le lancement doit réessayer.
bool is_postponed(const core::Settings& engines, core::EngineId id);

// La marque d'un essai, « <dossier>/<moteur>.retry » : un essai raté ne se refait pas avant retry_after,
// sinon chaque lancement hors ligne montrerait un écran d'installation qui échoue.
inline constexpr std::chrono::hours kRetryInstallAfter{6};
std::filesystem::path retry_marker(const std::filesystem::path& engines_root, core::EngineId id);
bool retry_install_due(const std::filesystem::path& marker);
void note_install_attempt(const std::filesystem::path& marker);

}  // namespace ajp::app::play
