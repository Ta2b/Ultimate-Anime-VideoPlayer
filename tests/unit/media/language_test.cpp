// Fixe le comportement repris de mp_match_lang (the-database-mpv misc/language.c:252-299).
#include "media/language.h"

#include <climits>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using ajp::media::canonical_language;
using ajp::media::language_score;

TEST_CASE("language : ISO 639-1, 639-2/B et 639-2/T ont une même forme canonique", "[media][language]") {
  CHECK(canonical_language("ja") == "jpn");
  CHECK(canonical_language("jpn") == "jpn");
  CHECK(canonical_language("fr") == "fre");
  CHECK(canonical_language("fra") == "fre");
  CHECK(canonical_language("FR") == "fre");
  CHECK(canonical_language("deu") == "ger");
  CHECK(canonical_language("zho") == "chi");
  CHECK(canonical_language("xx") == "xx");  // inconnue : telle quelle
}

TEST_CASE("language : une langue plus haut dans la liste l'emporte", "[media][language]") {
  const std::vector<std::string> priorities{"jpn", "en"};
  CHECK(language_score(priorities, "ja") == INT_MAX);
  CHECK(language_score(priorities, "jpn") == INT_MAX);
  CHECK(language_score(priorities, "eng") == INT_MAX - 1);
  CHECK(language_score(priorities, "ja") > language_score(priorities, "en"));
  CHECK(language_score(priorities, "fre") == 0);
  CHECK(language_score(priorities, "") == 0);
}

TEST_CASE("language : chaque sous-balise différente ou non demandée coûte 1000", "[media][language]") {
  const std::vector<std::string> us{"en-US"};
  CHECK(language_score(us, "en-US") == INT_MAX);
  CHECK(language_score(us, "en-us") == INT_MAX);  // casse ignorée
  CHECK(language_score(us, "en-GB") == INT_MAX - 1000);
  CHECK(language_score(us, "en") == INT_MAX);  // sous-balise voulue absente : sans coût

  const std::vector<std::string> plain{"en"};
  CHECK(language_score(plain, "en-US") == INT_MAX - 1000);
  CHECK(language_score(plain, "en-US-x-a") == INT_MAX - 3000);

  // Une sous-balise différente coûte plus qu'un rang dans la liste.
  const std::vector<std::string> both{"en", "fr-CA"};
  CHECK(language_score(both, "fr-CA") == INT_MAX - 1);
  CHECK(language_score(both, "en-GB") < language_score(both, "fr-CA"));
  CHECK(language_score(both, "en") > language_score(both, "fr-CA"));
}

TEST_CASE("language : une priorité vide ne correspond à rien", "[media][language]") {
  const std::vector<std::string> empty_entry{""};
  CHECK(language_score(empty_entry, "jpn") == 0);
  CHECK(language_score({}, "jpn") == 0);
}
