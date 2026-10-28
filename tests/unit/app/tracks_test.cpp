// Libellés et ordre des pistes que la barre de lecture propose.
#include "app/play/tracks.h"

#include <catch2/catch_test_macros.hpp>

namespace ajp::app::play {
namespace {

media::TrackInfo track(int index, media::TrackKind kind, std::string codec, std::string language,
                       std::string title) {
  media::TrackInfo info;
  info.index = index;
  info.kind = kind;
  info.codec = std::move(codec);
  info.language = std::move(language);
  info.title = std::move(title);
  return info;
}

TEST_CASE("Les langues du conteneur ont leur nom français", "[app][tracks]") {
  CHECK(language_name("jpn") == "Japonais");
  CHECK(language_name("fre") == "Français");
  CHECK(language_name("fra") == "Français");
  CHECK(language_name("en-US") == "Anglais");
  CHECK(language_name("ES") == "Espagnol");
  CHECK(language_name("und").empty());
  CHECK(language_name("").empty());
  CHECK(language_name("tlh") == "tlh");  // inconnue : telle quelle
  // Toutes les formes ISO 639 d'une langue : une seule ligne de la table.
  CHECK(language_name("uk") == "Ukrainien");
  CHECK(language_name("ukr") == "Ukrainien");
  CHECK(language_name("zho") == "Chinois");
  CHECK(language_name("nld") == "Néerlandais");
  CHECK(language_name("pt_BR") == "Portugais");
}

TEST_CASE("Les menus des langues préférées proposent les langues marquées, dans l'ordre", "[app][tracks]") {
  const std::span<const Language> menu = menu_languages();
  REQUIRE(menu.size() == 11);
  CHECK(menu.front().code == "jpn");
  CHECK(menu[1].code == "fre");
  CHECK(menu.back().code == "ara");
  for (const Language& language : menu) {
    CHECK(language.menu);
    CHECK(language_name(language.code) == language.name);
  }
  CHECK(same_language("fra", "fre"));
  CHECK(same_language("fr-CA", "FRE"));
  CHECK(same_language("ja", "jpn"));
  CHECK_FALSE(same_language("jpn", "chi"));
}

TEST_CASE("Le libellé d'une piste dit sa langue et son titre", "[app][tracks]") {
  using media::TrackKind;
  CHECK(track_label(track(2, TrackKind::Subtitle, "ass", "fre", "CR_French")) == "Français · CR_French");
  CHECK(track_label(track(1, TrackKind::Audio, "aac", "jpn", "")) == "Japonais");
  CHECK(track_label(track(3, TrackKind::Subtitle, "ass", "", "Signs")) == "Signs");
  CHECK(track_label(track(4, TrackKind::Subtitle, "subrip", "eng", "")) == "Anglais (SRT)");
  CHECK(track_label(track(5, TrackKind::Audio, "flac", "", "")) == "Piste 5");
  CHECK(track_label(track(6, TrackKind::Audio, "aac", "fre", "Français 5.1")) ==
        "Français 5.1");  // pas répété
  auto forced = track(7, TrackKind::Subtitle, "ass", "fre", "");
  forced.forced = true;
  CHECK(track_label(forced) == "Français · forcés");
}

TEST_CASE("j fait le tour des sous-titres puis aucun ; a fait le tour de l'audio", "[app][tracks]") {
  TrackMenu menu{.audio = {{.index = 1, .label = "Japonais"}, {.index = 2, .label = "Anglais"}},
                 .subtitles = {{.index = 3, .label = "Français"}, {.index = 4, .label = "Anglais"}},
                 .audio_selected = 1,
                 .subtitle_selected = 3};
  CHECK(next_subtitle(menu) == 4);
  menu.subtitle_selected = 4;
  CHECK_FALSE(next_subtitle(menu));
  menu.subtitle_selected.reset();
  CHECK(next_subtitle(menu) == 3);
  CHECK(next_audio(menu) == 2);
  menu.audio_selected = 2;
  CHECK(next_audio(menu) == 1);
  CHECK_FALSE(next_audio(TrackMenu{}));
  CHECK_FALSE(next_subtitle(TrackMenu{}));
}

}  // namespace
}  // namespace ajp::app::play
