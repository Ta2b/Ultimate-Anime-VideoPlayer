// Le fichier du journal (ajp.log) : ouvert par son chemin, donc en UTF-16 sous Windows. spdlog, construit
// sans wchar_t, n'ouvre qu'un nom en page de code ANSI : un dossier de données au nom japonais ou russe sur
// un Windows français ne s'ouvrait pas, et le journal partait sans un mot (B33-06).
#pragma once

#include <cstddef>
#include <filesystem>

#include <spdlog/common.h>

namespace ajp {

// Un puits qui écrit dans file, ouvert en ajout. Passé max_bytes, file devient <nom>.1<ext>, l'ancien
// <nom>.1<ext> devient <nom>.2<ext>, et ainsi de suite jusqu'à parts (au-delà : effacé), comme le
// rotating_file_sink de spdlog. nullptr si le fichier ne s'ouvre pas.
spdlog::sink_ptr log_file_sink(const std::filesystem::path& file, std::size_t max_bytes, std::size_t parts);

// Le nom de la part index de file : « ajp.log », « ajp.1.log », « ajp.2.log ».
std::filesystem::path log_part(const std::filesystem::path& file, std::size_t index);

// Au lancement : les journaux des kept séances d'avant restent (ajp.log.1 le plus récent) ; après un gel ou
// un arrêt brutal, la relance n'efface plus celui qui en garde la trace. Les parts d'une séance trop longue
// (ajp.1.log, ajp.2.log, jusqu'à parts) tournent avec leur journal (A33-06) : ajp.1.log.1 est de la même
// séance que ajp.log.1.
void rotate_session_logs(const std::filesystem::path& file, int kept, std::size_t parts);

}  // namespace ajp
