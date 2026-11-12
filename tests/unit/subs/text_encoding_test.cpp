#include "subs/text_encoding.h"

#include <string>

#include <catch2/catch_test_macros.hpp>

using ajp::subs::to_utf8;

TEST_CASE("to_utf8 : l'UTF-8 reste tel quel, BOM retiré", "[encoding]") {
  CHECK(to_utf8("caf\xC3\xA9") == "caf\xC3\xA9");
  CHECK(to_utf8("\xEF\xBB\xBF"
                "abc") == "abc");
  CHECK(to_utf8("") == "");
}

TEST_CASE("to_utf8 : UTF-16 des deux endianness, avec paire de substitution", "[encoding]") {
  CHECK(to_utf8(std::string_view("\xFF\xFE"
                                 "a\0\xE9\0",
                                 6)) == "a\xC3\xA9");
  CHECK(to_utf8(std::string_view("\xFE\xFF\0a\0\xE9", 6)) == "a\xC3\xA9");
  CHECK(to_utf8(std::string_view("\xFF\xFE\x3D\xD8\x00\xDE", 6)) == "\xF0\x9F\x98\x80");
}

TEST_CASE("to_utf8 : un texte qui n'est pas de l'UTF-8 est lu en Windows-1252", "[encoding]") {
  CHECK(to_utf8("caf\xE9") == "caf\xC3\xA9");
  CHECK(to_utf8("\x93quoi\x94") == "\xE2\x80\x9Cquoi\xE2\x80\x9D");
  CHECK(to_utf8("\x80") == "\xE2\x82\xAC");
}

TEST_CASE("to_utf8 : RFC 3629 et encodages hérités (P3.74)", "[encoding]") {
  // Formes trop longues et moitiés de paire : pas de l'UTF-8, lues en Windows-1252.
  CHECK(to_utf8("\xE0\x80\xAF") == "\xC3\xA0\xE2\x82\xAC\xC2\xAF");
  CHECK(to_utf8("\xED\xA0\x80") != "\xED\xA0\x80");
  // こんにちは en Shift-JIS, 你好世界 en GBK et en Big5, Привет мир в CP1251.
  const std::string japanese = "\x82\xb1\x82\xf1\x82\xc9\x82\xbf\x82\xcd";
  CHECK(to_utf8(japanese) == "\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf");
  CHECK(to_utf8("\xc4\xe3\xba\xc3\xca\xc0\xbd\xe7", "chs") ==
        "\xe4\xbd\xa0\xe5\xa5\xbd\xe4\xb8\x96\xe7\x95\x8c");
  CHECK(to_utf8("\xa7\x41\xa6\x6e\xa5\x40\xac\xc9") == "\xe4\xbd\xa0\xe5\xa5\xbd\xe4\xb8\x96\xe7\x95\x8c");
  // Extensions ETEN de CP950 (碁 en F9D6, € en A3E1), décodées comme Windows (A34-11).
  CHECK(to_utf8("\xa7\x41\xf9\xd6\xa3\xe1", "cht") == "\xe4\xbd\xa0\xe7\xa2\x81\xe2\x82\xac");
  CHECK(to_utf8("\xcf\xf0\xe8\xe2\xe5\xf2 \xec\xe8\xf0") ==
        "\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82 \xd0\xbc\xd0\xb8\xd1\x80");
  CHECK(to_utf8("caf\xE9 cr\xE8me") == "caf\xC3\xA9 cr\xC3\xA8me");
  CHECK(to_utf8("l'\xE9t\xE9 dernier, d\xE9j\xE0 vu \xE0 c\xF4t\xE9") ==
        "l'\xC3\xA9t\xC3\xA9 dernier, d\xC3\xA9j\xC3\xA0 vu \xC3\xA0 c\xC3\xB4t\xC3\xA9");
}

namespace {

// 안녕하세요, 여러분. 오늘 날씨가 정말 좋네요. en EUC-KR ; こんにちは、みなさん。今日はいい天気ですね。 en
// EUC-JP ; 你好，我們今天去看電影吧。 en Big5. Encodés par Python (« … ».encode('euc_kr'), 'euc_jp', 'big5').
const std::string kEucKr =
    "\xbe\xc8\xb3\xe7\xc7\xcf\xbc\xbc\xbf\xe4\x2c\x20\xbf\xa9\xb7\xaf\xba\xd0\x2e\x20\xbf\xc0\xb4\xc3\x20\xb3"
    "\xaf"
    "\xbe\xbe\xb0\xa1\x20\xc1\xa4\xb8\xbb\x20\xc1\xc1\xb3\xd7\xbf\xe4\x2e";
const std::string kKo =
    "\xec\x95\x88\xeb\x85\x95\xed\x95\x98\xec\x84\xb8\xec\x9a\x94\x2c\x20\xec\x97\xac\xeb\x9f\xac\xeb\xb6\x84"
    "\x2e"
    "\x20\xec\x98\xa4\xeb\x8a\x98\x20\xeb\x82\xa0\xec\x94\xa8\xea\xb0\x80\x20\xec\xa0\x95\xeb\xa7\x90\x20\xec"
    "\xa2"
    "\x8b\xeb\x84\xa4\xec\x9a\x94\x2e";
const std::string kEucJp =
    "\xa4\xb3\xa4\xf3\xa4\xcb\xa4\xc1\xa4\xcf\xa1\xa2\xa4\xdf\xa4\xca\xa4\xb5\xa4\xf3\xa1\xa3\xba\xa3\xc6\xfc"
    "\xa4"
    "\xcf\xa4\xa4\xa4\xa4\xc5\xb7\xb5\xa4\xa4\xc7\xa4\xb9\xa4\xcd\xa1\xa3";
const std::string kJa =
    "\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf\xe3\x80\x81\xe3\x81\xbf\xe3\x81\xaa\xe3\x81"
    "\x95"
    "\xe3\x82\x93\xe3\x80\x82\xe4\xbb\x8a\xe6\x97\xa5\xe3\x81\xaf\xe3\x81\x84\xe3\x81\x84\xe5\xa4\xa9\xe6\xb0"
    "\x97"
    "\xe3\x81\xa7\xe3\x81\x99\xe3\x81\xad\xe3\x80\x82";
const std::string kBig5 =
    "\xa7\x41\xa6\x6e\xa1\x41\xa7\xda\xad\xcc\xa4\xb5\xa4\xd1\xa5\x68\xac\xdd\xb9\x71\xbc\x76\xa7\x61\xa1"
    "\x43";
const std::string kTw =
    "\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c\xe6\x88\x91\xe5\x80\x91\xe4\xbb\x8a\xe5\xa4\xa9\xe5\x8e\xbb\xe7\x9c"
    "\x8b"
    "\xe9\x9b\xbb\xe5\xbd\xb1\xe5\x90\xa7\xe3\x80\x82";

// Un SRT d'une réplique, en UTF-16 sans BOM, à partir de son UTF-8 (BMP seulement).
std::string utf16_srt(const std::string& utf8_line, bool big_endian) {
  const std::u32string line = [&] {
    std::u32string out;
    for (std::size_t i = 0; i < utf8_line.size();) {
      const auto b = static_cast<unsigned char>(utf8_line[i]);
      const std::size_t n = b < 0x80 ? 1 : b < 0xE0 ? 2 : 3;
      char32_t code = n == 1 ? b : n == 2 ? b & 0x1F : b & 0x0F;
      for (std::size_t k = 1; k < n; ++k)
        code = (code << 6) | (static_cast<unsigned char>(utf8_line[i + k]) & 0x3F);
      out.push_back(code);
      i += n;
    }
    return out;
  }();
  const std::u32string text = U"1\n00:00:01,000 --> 00:00:03,000\n" + line + U"\n";
  std::string bytes;
  for (const char32_t c : text) {
    const auto hi = static_cast<char>(c >> 8);
    const auto lo = static_cast<char>(c & 0xFF);
    bytes += big_endian ? std::string{hi, lo} : std::string{lo, hi};
  }
  return bytes;
}

}  // namespace

TEST_CASE("to_utf8 : coréen EUC-KR, avec ou sans indice (A17-03)", "[encoding]") {
  CHECK(to_utf8(kEucKr) == kKo);
  CHECK(to_utf8(kEucKr, "ko") == kKo);
  CHECK(to_utf8(kEucKr, "kor.forced") == kKo);
}

TEST_CASE("to_utf8 : japonais EUC-JP, avec ou sans indice (A17-03)", "[encoding]") {
  CHECK(to_utf8(kEucJp) == kJa);
  CHECK(to_utf8(kEucJp, "ja") == kJa);  // l'indice « ja » propose Shift-JIS puis EUC-JP
  CHECK(to_utf8(kEucJp, "jpn") == kJa);
}

TEST_CASE("to_utf8 : un Big5 nommé .zh reste du Big5 (A17-03)", "[encoding]") {
  CHECK(to_utf8(kBig5) == kTw);
  CHECK(to_utf8(kBig5, "zh") == kTw);
  CHECK(to_utf8(kBig5, "zh_TW") == kTw);
  CHECK(to_utf8(kBig5, "chs&jpn.forced") == kTw);
}

TEST_CASE("to_utf8 : l'indice est lu dans tous les suffixes (A17-03)", "[encoding]") {
  const std::string gbk = "\xc4\xe3\xba\xc3\xca\xc0\xbd\xe7";  // 你好世界
  const std::string world = "\xe4\xbd\xa0\xe5\xa5\xbd\xe4\xb8\x96\xe7\x95\x8c";
  CHECK(to_utf8(gbk, "chs.forced") == world);
  CHECK(to_utf8(gbk, "chs&jpn") == world);
  CHECK(to_utf8(gbk, "zh_CN") == world);
}

TEST_CASE("to_utf8 : un indice qui ne décode pas laisse la forme des octets décider (A17-03)", "[encoding]") {
  // Un SRT Shift-JIS nommé .euc-jp : l'EUC-JP refuse les têtes 0x82, la forme des octets reprend la main.
  const std::string japanese = "\x82\xb1\x82\xf1\x82\xc9\x82\xbf\x82\xcd";
  CHECK(to_utf8(japanese, "euc-jp") == "\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf");
}

TEST_CASE("to_utf8 : UTF-16 sans BOM, little et big endian (A17-04)", "[encoding]") {
  CHECK(to_utf8(utf16_srt(kJa, false)).find(kJa) != std::string::npos);
  CHECK(to_utf8(utf16_srt(kJa, true)).find(kJa) != std::string::npos);
  CHECK(to_utf8(utf16_srt("caf\xc3\xa9", false)) == "1\n00:00:01,000 --> 00:00:03,000\ncaf\xc3\xa9\n");
}

TEST_CASE("to_utf8 : une demi-paire UTF-16 donne U+FFFD (A17-04)", "[encoding]") {
  const std::string lone("\xFF\xFE\x00\xD8\x41\x00", 6);  // BOM, D800 seul, 'A'
  CHECK(to_utf8(lone) ==
        "\xEF\xBF\xBD"
        "A");
  const std::string low("\xFF\xFE\x00\xDC", 4);  // DC00 seul
  CHECK(to_utf8(low) == "\xEF\xBF\xBD");
  const std::string cut("\xFF\xFE\x41\x00\x3D\xD8", 6);  // 'A', puis une tête de paire coupée
  CHECK(to_utf8(cut) == "A\xEF\xBF\xBD");
}

// B17-05 : le nom du fichier nomme une langue qui n'a pas son Windows-125x.
TEST_CASE("to_utf8 : les pages d'une seule langue, prises sur l'indice du nom", "[encoding]") {
  CHECK(to_utf8("za\xBF\xF3\xB3\xE6", "pl") == "za\xC5\xBC\xC3\xB3\xC5\x82\xC4\x87");  // zażółć, CP1250
  CHECK(to_utf8("\xDD\xFE\xFC\xF0\xFD\xFE", "tr") ==
        "\xC4\xB0\xC5\x9F\xC3\xBC\xC4\x9F\xC4\xB1\xC5\x9F");  // CP1254
  CHECK(to_utf8("\xE1\xE2", "el") == "\xCE\xB1\xCE\xB2");     // αβ, CP1253
}

TEST_CASE("to_utf8 : un octet sans caractère d'une page à un octet ne fait pas perdre le texte (C34-07)",
          "[encoding]") {
  // « Привет » en CP1251, puis 0x98, case non définie de CP1251 : avant, iconv refusait tout et le texte
  // retombait en Windows-1252 (« Ïðèâåò˜ ») sous Linux seulement.
  const std::string bytes = "\xCF\xF0\xE8\xE2\xE5\xF2\x98";
  CHECK(ajp::subs::to_utf8(bytes, "ep01.ru.srt").starts_with("Привет"));
  // Grec (CP1253, 0xAA non défini) : le reste du texte est gardé.
  CHECK(ajp::subs::to_utf8("\xE3\xE5\xE9\xDC\xAA", "ep01.el.srt").starts_with("γειά"));
}
