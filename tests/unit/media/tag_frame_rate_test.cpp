#include <string>

#include <catch2/catch_test_macros.hpp>

#include "media/frame_rate.h"
#include "media/tag_text.h"

using ajp::media::clean_tag_text;
using ajp::media::Rational;
using ajp::media::standard_frame_rate;

// C14-07 : une moyenne estimée par le conteneur retombe sur la cadence de diffusion.
TEST_CASE("standard_frame_rate : les moyennes estimées retrouvent leur cadence", "[media][frame_rate]") {
  CHECK(standard_frame_rate({2997, 100}) == Rational{30000, 1001});
  CHECK(standard_frame_rate({29970, 1000}) == Rational{30000, 1001});
  CHECK(standard_frame_rate({23976, 1000}) == Rational{24000, 1001});
  CHECK(standard_frame_rate({23977, 1000}) == Rational{24000, 1001});
  CHECK(standard_frame_rate({24, 1}) == Rational{24, 1});
  CHECK(standard_frame_rate({10000, 400}) == Rational{25, 1});
  CHECK(standard_frame_rate({119880, 1000}) == Rational{120000, 1001});
}

TEST_CASE("standard_frame_rate : une cadence éloignée ou invalide reste telle quelle",
          "[media][frame_rate]") {
  CHECK(standard_frame_rate({2350, 100}) == Rational{2350, 100});  // VFR à 23,5 i/s
  CHECK(standard_frame_rate({15, 1}) == Rational{15, 1});
  CHECK(standard_frame_rate({}) == Rational{});
  CHECK(standard_frame_rate({0, 1}) == Rational{0, 1});
}

// C14-08 : un titre de piste vient du fichier.
TEST_CASE("clean_tag_text : UTF-8 réparé, contrôles et marques bidirectionnelles retirés", "[media][tag]") {
  CHECK(clean_tag_text("Japonais 5.1") == "Japonais 5.1");
  CHECK(clean_tag_text("Fran\xC3\xA7"
                       "ais") ==
        "Fran\xC3\xA7"
        "ais");
  CHECK(clean_tag_text("Fran\xE7"
                       "ais") ==
        "Fran\xEF\xBF\xBD"
        "ais");  // Latin-1 mal étiqueté
  CHECK(clean_tag_text("a\nb\tc") == "a b c");
  CHECK(clean_tag_text("avant\xE2\x80\xAE"
                       "apres") == "avantapres");               // U+202E
  CHECK(clean_tag_text("x\xE2\x81\xA6y\xE2\x81\xA9") == "xy");  // U+2066, U+2069
  CHECK(clean_tag_text("\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E") == "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E");
}

TEST_CASE("clean_tag_text : coupé à 256 octets, jamais au milieu d'un caractère", "[media][tag]") {
  std::string text;
  for (int i = 0; i < 200; ++i) text += "\xE6\x97\xA5";  // 600 octets
  const std::string cut = clean_tag_text(text);
  CHECK(cut.size() == 255);  // 85 caractères de 3 octets
  CHECK(clean_tag_text(std::string(1000, 'a')).size() == ajp::media::kMaxTagBytes);
}
