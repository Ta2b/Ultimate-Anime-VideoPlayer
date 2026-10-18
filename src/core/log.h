// Journal du lecteur : spdlog, un canal nommé par module, des puits installés par app.
//
// Convention, la même pour tous les modules :
//   - un module journalise par log_channel("<module>"), jamais par le journal par défaut
//     ni par le registre global de spdlog ; son en-tête log.h ne fait que nommer le canal ;
//   - app appelle install_log_sinks() au démarrage, depuis main, avant de lancer les
//     threads ; tous les canaux créés ensuite partagent un même répartiteur, si bien qu'un
//     puits installé plus tard (fichier, OSD) reçoit aussi leurs messages ;
//   - le niveau d'un canal est celui du journal par défaut au moment de sa création ;
//   - un backend chargé à l'exécution a sa propre copie de spdlog : il journalise par le
//     canal que lui passe BackendInit::log (gpu/backend.h), pas par log_channel().
// Cible ajp::core_log : dépend de spdlog seulement, en dépendance privée des modules.
#pragma once

#include <memory>
#include <string_view>
#include <vector>

#include <spdlog/common.h>
#include <spdlog/logger.h>

namespace ajp {

// Canal du module : les puits du journal par défaut au moment de l'appel, partagés et non
// copiés, son niveau et son niveau de vidage. Le canal n'entre pas au registre de spdlog.
std::shared_ptr<spdlog::logger> log_channel(std::string_view module);

// Remplace les puits de tous les canaux par sinks, au niveau level. Le premier appel pose un
// répartiteur unique dans le journal par défaut ; les suivants ne changent que son contenu.
void install_log_sinks(std::vector<spdlog::sink_ptr> sinks, spdlog::level::level_enum level);

}  // namespace ajp
