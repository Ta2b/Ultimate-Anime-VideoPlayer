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
