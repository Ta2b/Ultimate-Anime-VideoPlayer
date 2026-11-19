// L'arbre des réglages (settings_panel.h : settings_pages) : sa taille, sa cohérence, et les renvois que les
// autres écrans (assistant de premier démarrage) y font.
#include <algorithm>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/french_text.h"
#include "app/play/i18n.h"
#include "app/play/loading_settings.h"
#include "app/play/panel_tip.h"
#include "app/play/settings_panel.h"
#include "app/play/wizard.h"

using namespace kirari::app::play;

TEST_CASE("le menu des réglages tient en cinq catégories, Général d'abord", "[settings][pages]") {
  const auto pages = settings_pages();
  CHECK(pages.size() <= 10);
  std::vector<std::string_view> categories;
  std::set<std::pair<std::string_view, std::string_view>> names;
  std::string_view previous;
  for (const SettingsPage& page : pages) {
    CHECK_FALSE(std::string_view(page.name).empty());
    CHECK(
        names.insert({page.category, page.name}).second);  // un onglet n'est donné qu'une fois par catégorie
    // Les onglets d'une catégorie se suivent : le panneau déduit les catégories du changement de `category`.
    if (page.category != previous) {
      CHECK(std::ranges::find(categories, std::string_view(page.category)) == categories.end());
      categories.emplace_back(page.category);
    }
    previous = page.category;
  }
  CHECK(categories == std::vector<std::string_view>{"General", "Video", "Audio", "Languages", "Export"});
  // Le panneau s'ouvre sur Vidéo > Général (settings_panel.h : page_ = 2).
  CHECK(pages[2].id == Page::VideoGeneral);
}

TEST_CASE("le menu des réglages garde les onglets que les autres écrans citent", "[settings][pages]") {
  const auto pages = settings_pages();
  const auto has = [&](std::string_view category, std::string_view name) {
    return std::ranges::any_of(pages, [&](const SettingsPage& page) {
      return std::string_view(page.category) == category && std::string_view(page.name) == name;
    });
  };
  CHECK(has("General", "Application"));
  CHECK(has("General", "Statistics"));
  CHECK(has("Video", "General"));  // le moteur et le préréglage
  CHECK(has("Video", "Upscaling"));
  CHECK(has("Video", "Interpolation"));
  CHECK(has("Video", "HDR"));
  CHECK(has("Languages", "Languages"));
  CHECK_FALSE(has("General", "Engines"));  // le moteur est passé dans Vidéo > Général
  CHECK_FALSE(has("General", "Export"));
}

TEST_CASE("le résumé de l'assistant renvoie à des pages qui existent", "[settings][pages][wizard]") {
  const auto pages = settings_pages();
  i18n::set_language(i18n::Language::French);  // le résumé de l'assistant est encore en français
  Wizard wizard = begin_wizard({.engine_choice = true}, 0);
  const auto lines = wizard_summary(wizard);
  REQUIRE(lines.size() >= 2);
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
    CHECK(std::ranges::any_of(pages, [&](const SettingsPage& page) {
      return category == i18n::tr(page.category) && name == i18n::tr(page.name);
    }));
  }
  i18n::set_language(i18n::Language::English);
}

TEST_CASE("les notes gardent les guillemets et la ponctuation haute avec leur mot", "[settings][typo]") {
  using kirari::app::play::french_spaces;
  CHECK(french_spaces("bouton « Préparer » plus bas : voir") ==
        "bouton «\xC2\xA0Préparer\xC2\xA0» plus bas\xC2\xA0: voir");
  CHECK(french_spaces("Quoi ? Oui ! 3 % ; fin") == "Quoi\xC2\xA0? Oui\xC2\xA0! 3\xC2\xA0%\xC2\xA0; fin");
  CHECK(french_spaces("rien à changer") == "rien à changer");
  CHECK(french_spaces("") == "");
}

TEST_CASE("une infobulle se découpe en description, vitesse, qualité et recommandation", "[settings][tip]") {
  using kirari::app::play::panel::split_tip;
  const auto parts = split_tip(
      "In-between frames read the source with a sharper filter. Speed: about −3 %. Quality impact: minor "
      "gain. "
      "Recommended: on. RIFE only.");
  CHECK(parts.text == "In-between frames read the source with a sharper filter.");
  CHECK(parts.speed == "about −3 %");
  CHECK(parts.quality == "minor gain");
  CHECK(parts.recommended == "on");
  CHECK(parts.after == "RIFE only.");
  // Les repères français, et un texte sans repère reste entier.
  const auto french = split_tip("Un filtre plus net. Vitesse : environ −3 %. Recommandé : activé.");
  CHECK(french.speed == "environ −3 %");
  CHECK(french.recommended == "activé");
  CHECK(split_tip("Une phrase seule.").text == "Une phrase seule.");
}

TEST_CASE("le menu pendant une préparation rouvre avec les réglages qu'il a confirmés",
          "[settings][loading]") {
  PlaySettings current;
  LoadingChange change;
  CHECK_FALSE(change.stops());
  PlaySettings live = current;
  live.look_contrast = 20;
  change.live = live;
  CHECK_FALSE(change.stops());  // un réglage en direct attend la lecture, sans arrêter la préparation
  CHECK(change.settings_or(current) == live);
  PlaySettings reopen = live;
  reopen.work_height = 540;
  change.reopen = reopen;
  CHECK(change.stops());
  CHECK(change.settings_or(current) == reopen);
}
