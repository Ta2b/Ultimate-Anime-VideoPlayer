// Le texte d'un fichier de sous-titres externe, ramené en UTF-8 : libass et l'analyseur SRT n'en
// lisent pas d'autre.
#pragma once

#include <string>
#include <string_view>

namespace ajp::subs {

// UTF-8 tel quel (BOM retiré) ; UTF-16 reconnu à son BOM (little ou big endian). Sinon (P3.74), un
// encodage hérité : celui que dit `hint` (la langue ou l'encodage du nom de fichier : « ja », « chs »,
// « cp1251 »…), à défaut celui que suggère la forme des octets (Shift-JIS, GBK, Big5, CP1251), décodé par
// le système ; le repli reste le Windows-1252 des SRT d'Europe de l'Ouest.
std::string to_utf8(std::string_view bytes, std::string_view hint = {});

}  // namespace ajp::subs
