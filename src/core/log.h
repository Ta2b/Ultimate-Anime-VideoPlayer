// Journal du lecteur : spdlog, un canal nommé par module, des puits installés par app.
//
// Convention, la même pour tous les modules :
//   - un module journalise par log_channel("<module>"), jamais par le journal par défaut
//     ni par le registre global de spdlog ; son en-tête log.h ne fait que nommer le canal ;
//   - app appelle install_log_sinks() au démarrage, depuis main, avant de lancer les
//     threads ; tous les canaux créés ensuite partagent un même répartiteur, si bien qu'un
//     puits installé plus tard (fichier, OSD) reçoit aussi leurs messages ;
//   - le niveau d'un canal est celui du journal par défaut au moment de sa création ;
//   - un backend chargé à l'exécution journalise par le rappel BackendInit::log
//     (gpu/backend.h), que app relie à un canal : ses propres canaux y vont par
//     install_module_log_sinks(), sans toucher au journal par défaut. Avec un spdlog partagé
//     (le spdlog.dll de vcpkg sous Windows), le registre et le journal par défaut sont ceux
//     du lecteur ; y poser le rappel le ferait se rappeler lui-même, sous le verrou du
//     répartiteur.
// Cible ajp::core_log : dépend de spdlog seulement, en dépendance privée des modules.
#pragma once

#include <memory>
#include <string_view>
#include <vector>

#include <spdlog/common.h>
#include <spdlog/logger.h>

namespace ajp {

// Canal du module : les puits du journal par défaut au moment de l'appel, partagés et non
// copiés, son niveau et son niveau de vidage ; ou, après install_module_log_sinks dans ce
// binaire, le répartiteur du binaire à son niveau. Le canal n'entre pas au registre de spdlog.
std::shared_ptr<spdlog::logger> log_channel(std::string_view module);

// Remplace les puits de tous les canaux par sinks, au niveau level. Le premier appel pose un
// répartiteur unique dans le journal par défaut ; les suivants ne changent que son contenu.
void install_log_sinks(std::vector<spdlog::sink_ptr> sinks, spdlog::level::level_enum level);

// Pour une bibliothèque chargée à l'exécution (un backend) : les canaux que ce binaire crée
// ensuite écrivent dans sinks, au niveau level, par un répartiteur propre au binaire (core_log
// est statique : chacun a le sien). Le journal par défaut et le registre de spdlog restent tels
// quels. Le premier appel pose le répartiteur ; les suivants ne changent que son contenu, que les
// canaux déjà créés suivent.
void install_module_log_sinks(std::vector<spdlog::sink_ptr> sinks, spdlog::level::level_enum level);

}  // namespace ajp
