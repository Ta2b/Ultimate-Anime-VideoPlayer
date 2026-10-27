// Le texte d'un fichier de sous-titres externe, ramené en UTF-8 : libass et l'analyseur SRT n'en
// lisent pas d'autre.
#pragma once

#include <string>
#include <string_view>

namespace ajp::subs {

// UTF-8 tel quel (BOM retiré) ; UTF-16 reconnu à son BOM (little ou big endian) ; sinon, un texte
// qui n'est pas de l'UTF-8 valide est lu comme du Windows-1252, l'encodage hérité le plus courant
// des SRT d'Europe de l'Ouest. Les encodages asiatiques (Shift-JIS, GBK…) ne sont pas reconnus :
// ils passent par le repli et donnent du texte illisible, sans refus.
std::string to_utf8(std::string_view bytes);

}  // namespace ajp::subs
