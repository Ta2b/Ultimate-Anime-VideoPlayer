// La langue du système : KDE Plasma (plasma-localerc, kdeglobals), le choix d'une langue des menus et les
// préférences de langue par défaut.
#include "app/play/system_language.h"

#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/settings.h"

TEST_CASE("langue du système : KDE Plasma la range dans plasma-localerc, sinon dans kdeglobals",
          "[app][language]") {
  using ajp::app::play::kde_locales;
  const std::string plasma = "[Formats]\nLANG=en_US.UTF-8\n\n[Translations]\nLANGUAGE=fr:en_US\n";
  CHECK(kde_locales(plasma, "") == std::vector<std::string>{"fr", "en_US"});
  CHECK(kde_locales("[Translations]\nLANGUAGE[$e]= de : en \n", "") == std::vector<std::string>{"de", "en"});
  // Seule la région posée ([Formats]) : ce n'est pas la langue de l'interface.
  CHECK(kde_locales("[Formats]\nLANG=fr_FR.UTF-8\n", "").empty());
  CHECK(kde_locales("", "[General]\nLanguage=it\n[Locale]\nLanguage=es\n") == std::vector<std::string>{"es"});
  CHECK(kde_locales("", "").empty());
}

TEST_CASE("langue du système : la première que les menus proposent, l'anglais sinon", "[app][language]") {
  using ajp::app::play::pick_menu_language;
  CHECK(pick_menu_language(std::vector<std::string>{"fr_FR.UTF-8"}) == "fre");
  CHECK(pick_menu_language(std::vector<std::string>{"C", "de-DE"}) == "ger");
  CHECK(pick_menu_language(std::vector<std::string>{"pt_BR"}) == "por");
  CHECK(pick_menu_language(std::vector<std::string>{"zh_TW"}) == "chi");
  CHECK(pick_menu_language(std::vector<std::string>{"ja"}) == "jpn");
  CHECK(pick_menu_language(std::vector<std::string>{"tlh", "it"}) == "ita");  // inconnue : la suivante
  CHECK(pick_menu_language(std::vector<std::string>{"uk_UA"}) == "eng");      // hors des menus : l'anglais
  CHECK(pick_menu_language(std::vector<std::string>{}) == "eng");
}

TEST_CASE("langues par défaut : audio japonais et sous-titres du système, puis audio du système",
          "[app][language]") {
  using namespace ajp::app::play;
  using Presets = std::vector<ajp::media::LanguagePreset>;
  set_system_language("fre");
  CHECK(default_language_presets() == Presets{{"jpn", "fre"}, {"fre", ""}});
  CHECK(PlaySettings{}.language_presets == Presets{{"jpn", "fre"}, {"fre", ""}});
  set_system_language("jpn");  // japonais : la version originale, sans sous-titres
  CHECK(default_language_presets() == Presets{{"jpn", ""}, {"", ""}});
  set_system_language("eng");
  CHECK(default_language_presets() == Presets{{"jpn", "eng"}, {"eng", ""}});
}

TEST_CASE("langue de l'interface : celle du système tant que play.toml n'en dit rien (A19-03)",
          "[app][language]") {
  using namespace ajp::app::play;
  set_system_language("fre");
  CHECK(PlaySettings{}.language == "fr");
  CHECK(parse_settings("")->language == "fr");
  CHECK(parse_settings("[interface]\nlanguage = \"en\"\n")->language == "en");  // un choix gardé l'emporte
  set_system_language("jpn");
  CHECK(PlaySettings{}.language == "en");
  set_system_language("eng");
  CHECK(PlaySettings{}.language == "en");
}
