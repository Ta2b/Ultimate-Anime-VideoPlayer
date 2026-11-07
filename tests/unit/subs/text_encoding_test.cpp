#include "subs/text_encoding.h"

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
  CHECK(to_utf8("\xcf\xf0\xe8\xe2\xe5\xf2 \xec\xe8\xf0") ==
        "\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82 \xd0\xbc\xd0\xb8\xd1\x80");
  CHECK(to_utf8("caf\xE9 cr\xE8me") == "caf\xC3\xA9 cr\xC3\xA8me");
  CHECK(to_utf8("l'\xE9t\xE9 dernier, d\xE9j\xE0 vu \xE0 c\xF4t\xE9") ==
        "l'\xC3\xA9t\xC3\xA9 dernier, d\xC3\xA9j\xC3\xA0 vu \xC3\xA0 c\xC3\xB4t\xC3\xA9");
}
