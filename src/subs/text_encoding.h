// Le texte d'un fichier de sous-titres externe, ramené en UTF-8 : libass et l'analyseur SRT n'en
// lisent pas d'autre.
#pragma once

#include <string>
#include <string_view>

namespace ajp::subs {

// UTF-8 tel quel (BOM retiré) ; UTF-16 reconnu à son BOM ou, sans BOM, à ses octets nuls alternés (une
// demi-paire seule donne U+FFFD). Sinon (P3.74), un encodage hérité : `hint` porte les suffixes du nom de
// fichier (« ja », « chs&jpn.forced », « zh_TW »…), lus du dernier au premier ; une langue propose ses
// encodages, entre lesquels la forme des octets choisit (Shift-JIS, EUC-JP, GBK, Big5, EUC-KR, CP1251), et
// un indice qui ne décode pas laisse la forme des octets décider. Le repli reste le Windows-1252 des SRT
// d'Europe de l'Ouest.
std::string to_utf8(std::string_view bytes, std::string_view hint = {});

}  // namespace ajp::subs
