// La langue lue dans le titre d'une piste sans balise, et son effet sur le choix des pistes.
#include "media/track_language.h"

#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "media/selection.h"

using ajp::media::choose_tracks;
using ajp::media::language_from_title;
using ajp::media::LanguagePriorities;
using ajp::media::track_language;
using ajp::media::TrackInfo;
using ajp::media::TrackKind;

namespace {

TrackInfo track(int index, TrackKind kind, const char* language, const char* title) {
  return TrackInfo{.index = index, .kind = kind, .language = language, .title = title};
}

}  // namespace

TEST_CASE("titre : les sigles et les noms de langue", "[media][language]") {
  CHECK(language_from_title("VOSTFR") == "fre");
  CHECK(language_from_title("[VF] 5.1") == "fre");
  CHECK(language_from_title("Français") == "fre");
  CHECK(language_from_title("FRANÇAIS") == "fre");
  CHECK(language_from_title("Japanese 2.0") == "jpn");
  CHECK(language_from_title("日本語") == "jpn");
  CHECK(language_from_title("日本語字幕") == "jpn");
  CHECK(language_from_title("English Signs & Songs") == "eng");
  CHECK(language_from_title("Español (Latino)") == "spa");
  CHECK(language_from_title("Русский") == "rus");
  CHECK(language_from_title("简体中文") == "chi");
}

TEST_CASE("titre : la première langue nommée, pas les mots ambigus", "[media][language]") {
  CHECK(language_from_title("Japanese [English subs]") == "jpn");
  CHECK(language_from_title("Sous-titres en français") == "fre");  // « en » ne compte pas
  CHECK(language_from_title("Commentaires") == "");
  CHECK(language_from_title("Signs").empty());
  CHECK(language_from_title("").empty());
  CHECK(language_from_title("Frenchie").empty());  // mot entier seulement
}

TEST_CASE("piste : la balise d'abord, le titre sans elle", "[media][language]") {
  CHECK(track_language(track(0, TrackKind::Audio, "jpn", "VF")) == "jpn");
  CHECK(track_language(track(0, TrackKind::Audio, "", "VF")) == "fre");
  CHECK(track_language(track(0, TrackKind::Audio, "und", "Japanese")) == "jpn");
  CHECK(track_language(track(0, TrackKind::Audio, "UND", "Japanese")) == "jpn");
  CHECK(track_language(track(0, TrackKind::Audio, "mul", "English")) == "eng");
  CHECK(track_language(track(0, TrackKind::Audio, "", "Stereo")).empty());
}

TEST_CASE("choix : les pistes sans balise suivent leur titre", "[media][language]") {
  const std::vector<TrackInfo> tracks{track(0, TrackKind::Video, "", ""),
                                      track(1, TrackKind::Audio, "", "VF"),
                                      track(2, TrackKind::Audio, "", "VO Japonais"),
                                      track(3, TrackKind::Subtitle, "", "Panneaux"),
                                      track(4, TrackKind::Subtitle, "", "VOSTFR"),
                                      track(5, TrackKind::Subtitle, "und", "English")};
  const LanguagePriorities fr{.audio = {"jpn"}, .subtitles = {"fre", "eng"}};
  const auto choice = choose_tracks(tracks, fr);
  CHECK(choice.audio == 2);
  CHECK(choice.subtitle == 4);
  const LanguagePriorities en{.audio = {"fre"}, .subtitles = {"eng"}};
  const auto english = choose_tracks(tracks, en);
  CHECK(english.audio == 1);
  CHECK(english.subtitle == 5);
}
