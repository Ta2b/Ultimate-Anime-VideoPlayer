// L'arbre des réglages (settings_panel.h : settings_pages) : sa taille, sa cohérence, et les renvois que les
// autres écrans (assistant de premier démarrage) y font.
#include <algorithm>
#include <set>
#include <string>
#include <string_view>

#include <catch2/catch_test_macros.hpp>

#include "app/play/french_text.h"
#include "app/play/settings_panel.h"
#include "app/play/wizard.h"

using namespace ajp::app::play;

TEST_CASE("l'arbre des réglages tient en cinq catégories et douze pages", "[settings][pages]") {
  const auto pages = settings_pages();
  CHECK(pages.size() <= 12);
  std::set<std::string_view> categories;
  std::set<std::string_view> names;
  std::string_view previous;
  for (const SettingsPage& page : pages) {
    CHECK_FALSE(std::string_view(page.name).empty());
    CHECK(names.insert(page.name).second);  // un nom de page n'est donné qu'une fois
    // Les pages d'une catégorie se suivent : l'arbre déduit les catégories du changement de `category`.
    if (page.category != previous) CHECK(categories.insert(page.category).second);
    previous = page.category;
  }
  CHECK(categories.size() <= 5);
  CHECK(std::string_view(pages.front().category) == "Général");
  CHECK(std::string_view(pages.front().name) == "Configuration");
}

TEST_CASE("l'arbre des réglages garde les pages que les autres écrans citent", "[settings][pages]") {
  const auto pages = settings_pages();
  const auto has = [&](std::string_view category, std::string_view name) {
    return std::ranges::any_of(pages, [&](const SettingsPage& page) {
      return std::string_view(page.category) == category && std::string_view(page.name) == name;
    });
  };
  // Les langues et les sous-titres, le lecteur par défaut et les mises à jour vivent ensemble.
  CHECK(has("Général", "Lecture"));
  CHECK(has("Général", "Application"));
  CHECK_FALSE(has("Général", "Langues"));
  CHECK_FALSE(has("Général", "Sous-titres"));
  CHECK_FALSE(has("Général", "Lecteur par défaut"));
  CHECK_FALSE(has("Général", "Mises à jour"));
}

TEST_CASE("le résumé de l'assistant renvoie à des pages qui existent", "[settings][pages][wizard]") {
  const auto pages = settings_pages();
  Wizard wizard = begin_wizard({.engine_choice = true, .hdr_display = true}, 0);
  const auto lines = wizard_summary(wizard);
  REQUIRE(lines.size() >= 5);
  for (const std::string& line : lines) {
    const std::size_t open = line.rfind('(');
    const std::size_t close = line.rfind(')');
    REQUIRE(open != std::string::npos);
    REQUIRE(close > open);
    const std::string path = line.substr(open + 1, close - open - 1);
    const std::size_t arrow = path.find(" > ");
    REQUIRE(arrow != std::string::npos);  // « Catégorie > Page »
    const std::string category = path.substr(0, arrow);
    const std::string name = path.substr(arrow + 3);
    INFO(line);
    CHECK(std::ranges::any_of(
        pages, [&](const SettingsPage& page) { return category == page.category && name == page.name; }));
  }
}

TEST_CASE("les notes gardent les guillemets et la ponctuation haute avec leur mot", "[settings][typo]") {
  using ajp::app::play::french_spaces;
  CHECK(french_spaces("bouton « Préparer » plus bas : voir") ==
        "bouton «\xC2\xA0Préparer\xC2\xA0» plus bas\xC2\xA0: voir");
  CHECK(french_spaces("Quoi ? Oui ! 3 % ; fin") == "Quoi\xC2\xA0? Oui\xC2\xA0! 3\xC2\xA0%\xC2\xA0; fin");
  CHECK(french_spaces("rien à changer") == "rien à changer");
  CHECK(french_spaces("") == "");
}
