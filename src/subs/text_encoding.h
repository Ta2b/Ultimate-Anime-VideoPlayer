// Le texte d'un fichier de sous-titres externe, ramené en UTF-8 : libass et l'analyseur SRT n'en
// lisent pas d'autre.
#pragma once

#include <optional>
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

// Le caractère d'un octet d'une page à un octet (CP125x, 874), tel que le système l'a décodé (rien : refusé).
// Une case sans caractère devient U+FFFD : refusée, rangée en zone privée (U+E000 à U+F8FF) ou rendue en
// commande C1 (U+0080 à U+009F) pour un octet de 0x80 à 0xFF, comme Windows le fait de ses cases non
// définies. Sous Linux, iconv les refuse et le décodage les remplace aussi par U+FFFD (C34-07). Pur.
char32_t single_byte_char(unsigned char byte, std::optional<char32_t> decoded);

}  // namespace ajp::subs
