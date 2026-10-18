// Chemins et textes UTF-8 : ce que Windows lirait autrement dans sa page ANSI.
#include "prepare/utf8.h"

#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "prepare/identity.h"
#include "support.h"

using namespace ajp::prepare;

TEST_CASE("is_utf8 : formes valides et mal formées", "[prepare][utf8]") {
  CHECK(is_utf8(""));
  CHECK(is_utf8("ASCII"));
  CHECK(is_utf8("é ü 日本 \xf0\x9f\x8e\x9e"));  // 2, 3 et 4 octets
  CHECK(is_utf8("\xf4\x8f\xbf\xbf"));           // U+10FFFF, le dernier point de code
  CHECK_FALSE(is_utf8("\xff"));                 // octet jamais valide
  CHECK_FALSE(is_utf8("\x80"));                 // octet de suite isolé
  CHECK_FALSE(is_utf8("\xc3"));                 // séquence coupée
  CHECK_FALSE(is_utf8("\xc0\xaf"));             // forme trop longue de « / »
  CHECK_FALSE(is_utf8("\xe0\x80\xaf"));         // idem sur trois octets
  CHECK_FALSE(is_utf8("\xed\xa0\x80"));         // moitié de paire UTF-16
  CHECK_FALSE(is_utf8("\xf4\x90\x80\x80"));     // au-delà de U+10FFFF
  CHECK_FALSE(is_utf8("\xf5\x80\x80\x80"));     // octet de tête hors plage
}

TEST_CASE("utf8_path et path_from_utf8 : aller-retour sans perte", "[prepare][utf8]") {
  const std::string text = "moteurs/日本 é/aji-v3-x.engine";
  const auto path = path_from_utf8(text);
  CHECK(utf8_path(path) == text);
  CHECK(utf8_path(path.filename()) == "aji-v3-x.engine");
  CHECK(path.u8string() == std::u8string(u8"moteurs/日本 é/aji-v3-x.engine"));
}

TEST_CASE("onnx_source_key : emplacement de poids non ASCII, comme fs::u8path du fork",
          "[prepare][utf8][identity]") {
  // Valeur calculée par content_identity.h et model_identity.cpp:20-27 du fork
  // (118c272), avec fs::u8path pour l'emplacement.
  const test::TempDir dir;
  const std::string location = "poids/é 日本.bin";
  test::write_file(dir.path() / path_from_utf8(location), "poids É");
  const std::vector<std::string> locations{location};
  CHECK(onnx_source_key(test::as_bytes("graphe ONNX factice"), dir.path(), locations).value() ==
        "e6a98b327e65651e802f17f4f369dce23d081ecd208906c22d895d584bfd66a5");
}
