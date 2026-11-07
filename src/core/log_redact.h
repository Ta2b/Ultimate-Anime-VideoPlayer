// Le journal nettoyé (P1.32) : ajp.log se joint à un rapport public, il ne doit dire ni le nom de
// l'utilisateur ni où sont ses vidéos. Le dossier personnel devient « ~ » dans chaque ligne, à
// l'entrée des puits ; les vidéos, elles, sont journalisées par leur nom de fichier seul.
#pragma once

#include <string>
#include <string_view>

#include <spdlog/common.h>

namespace ajp {

// Le dossier personnel, en UTF-8, tel que le système le donne. Vide : rien n'est remplacé.
void set_log_home(std::string_view home);

// text, le dossier personnel remplacé par « ~ », sous ses deux séparateurs (« \ » et « / »).
std::string redact_log_text(std::string_view text);

// Un puits qui écrit dans inner chaque ligne passée par redact_log_text.
spdlog::sink_ptr redacting_sink(spdlog::sink_ptr inner);

}  // namespace ajp
