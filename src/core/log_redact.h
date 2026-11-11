// Le journal nettoyé (P1.32) : ajp.log se joint à un rapport public, il ne doit dire ni le nom de
// l'utilisateur ni où sont ses vidéos. Le dossier personnel devient « ~ » dans chaque ligne, à
// l'entrée des puits, puis tout chemin absolu est réduit au nom de son fichier.
#pragma once

#include <string>
#include <string_view>

#include <spdlog/common.h>

namespace ajp {

// Le dossier personnel, en UTF-8, tel que le système le donne. Vide : rien n'est remplacé.
void set_log_home(std::string_view home);

// text, le dossier personnel remplacé par « ~ », sous ses deux séparateurs (« \ » et « / »), à une
// frontière de composant seulement (« /home/al » ne touche pas « /home/alice »), sans égard à la casse sous
// Windows et aussi sous sa forme courte 8.3 ; le nom de l'utilisateur sous /run/media/ et /media/ devient
// « <user> » (A20-03).
std::string redact_log_text(std::string_view text);

// text sans les dossiers : tout chemin absolu (Windows « D:\… », UNC, POSIX « /… », « ~/… ») est réduit à
// « …/<nom du fichier> ». Le reste de la ligne après le nom est gardé. Le rapport de « Signaler un problème »
// et le fichier ajp.log passent par là.
std::string reduce_log_paths(std::string_view text);

// Un puits qui écrit dans inner chaque ligne passée par redact_log_text puis reduce_log_paths : ajp.log est
// aussi propre que le rapport (B39-05).
spdlog::sink_ptr redacting_sink(spdlog::sink_ptr inner);

}  // namespace ajp
