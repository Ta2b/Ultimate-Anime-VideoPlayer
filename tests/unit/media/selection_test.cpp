// Fixe les règles reprises de compare_track et select_default_track
// (the-database-mpv player/loadfile.c:561-733).
#include "media/selection.h"

#include <vector>

#include <catch2/catch_test_macros.hpp>

using ajp::media::choose_tracks;
using ajp::media::LanguagePriorities;
using ajp::media::TrackInfo;
using ajp::media::TrackKind;

namespace {

TrackInfo track(int index, TrackKind kind, const char* language, bool is_default = false,
                bool forced = false) {
  return TrackInfo{
      .index = index, .kind = kind, .language = language, .is_default = is_default, .forced = forced};
}

}  // namespace

TEST_CASE("selection : la vidéo par défaut, jamais une pochette", "[media][selection]") {
  TrackInfo cover = track(0, TrackKind::Video, "", true);
  cover.attached_picture = true;
  const std::vector<TrackInfo> tracks{cover, track(1, TrackKind::Video, ""),
                                      track(2, TrackKind::Video, "", true)};
  CHECK(choose_tracks(tracks, {}).video == 2);

  const std::vector<TrackInfo> no_default{cover, track(3, TrackKind::Video, ""),
                                          track(4, TrackKind::Video, "")};
  CHECK(choose_tracks(no_default, {}).video == 3);
}

TEST_CASE("selection : l'audio suit la langue avant le drapeau défaut", "[media][selection]") {
  const std::vector<TrackInfo> tracks{track(1, TrackKind::Audio, "eng", true),
                                      track(2, TrackKind::Audio, "jpn"), track(3, TrackKind::Audio, "fre")};
  CHECK(choose_tracks(tracks, LanguagePriorities{.audio = {"ja", "en"}}).audio == 2);
  CHECK(choose_tracks(tracks, LanguagePriorities{.audio = {"fr"}}).audio == 3);
  CHECK(choose_tracks(tracks, LanguagePriorities{.audio = {"de"}}).audio == 1);  // défaut
  const std::vector<TrackInfo> plain{track(5, TrackKind::Audio, ""), track(6, TrackKind::Audio, "")};
  CHECK(choose_tracks(plain, {}).audio == 5);  // ordre du fichier
}

TEST_CASE("selection : sous-titres complets avant panneaux forcés", "[media][selection]") {
  const std::vector<TrackInfo> tracks{
      track(1, TrackKind::Audio, "jpn"), track(2, TrackKind::Subtitle, "fre", false, true),
      track(3, TrackKind::Subtitle, "fre"), track(4, TrackKind::Subtitle, "eng", true)};
  const LanguagePriorities fr{.audio = {"jpn"}, .subtitles = {"fr", "en"}};
  CHECK(choose_tracks(tracks, fr).subtitle == 3);
  const LanguagePriorities en{.audio = {"jpn"}, .subtitles = {"en"}};
  CHECK(choose_tracks(tracks, en).subtitle == 4);
}

TEST_CASE("selection : sans langue voulue, panneaux de la langue de l'audio, puis défaut",
          "[media][selection]") {
  const std::vector<TrackInfo> tracks{track(1, TrackKind::Audio, "eng"),
                                      track(2, TrackKind::Subtitle, "eng", false, true),
                                      track(3, TrackKind::Subtitle, "ger", true)};
  const LanguagePriorities wants_fr{.subtitles = {"fr"}};
  CHECK(choose_tracks(tracks, wants_fr).subtitle == 2);

  const std::vector<TrackInfo> no_forced{track(1, TrackKind::Audio, "eng"),
                                         track(3, TrackKind::Subtitle, "ger", true)};
  CHECK(choose_tracks(no_forced, wants_fr).subtitle == 3);

  const std::vector<TrackInfo> nothing{track(1, TrackKind::Audio, "eng"),
                                       track(3, TrackKind::Subtitle, "ger")};
  CHECK_FALSE(choose_tracks(nothing, wants_fr).subtitle.has_value());
  CHECK_FALSE(choose_tracks(nothing, wants_fr).video.has_value());
}

TEST_CASE("selection : la piste de dialogue avant Signs & Songs, commentaire et SDH", "[media][selection]") {
  TrackInfo signs = track(2, TrackKind::Subtitle, "eng", true);
  signs.title = "Signs & Songs";
  TrackInfo commentary = track(3, TrackKind::Subtitle, "eng");
  commentary.title = "Director's Commentary";
  TrackInfo sdh = track(4, TrackKind::Subtitle, "eng");
  sdh.title = "English [SDH]";
  TrackInfo full = track(5, TrackKind::Subtitle, "eng");
  full.title = "Full Subtitles";
  const LanguagePriorities en{.audio = {"jpn"}, .subtitles = {"en"}};

  const std::vector<TrackInfo> tracks{track(1, TrackKind::Audio, "jpn"), signs, commentary, sdh, full};
  CHECK(choose_tracks(tracks, en).subtitle == 5);

  // Faute de mieux, une piste spéciale vaut mieux que rien.
  const std::vector<TrackInfo> only_signs{track(1, TrackKind::Audio, "jpn"), signs, commentary};
  CHECK(choose_tracks(only_signs, en).subtitle == 2);
}

TEST_CASE("selection : les sous-titres image ne sont jamais choisis", "[media][selection]") {
  TrackInfo pgs = track(2, TrackKind::Subtitle, "eng", true);
  pgs.codec = "hdmv_pgs_subtitle";
  TrackInfo ass = track(3, TrackKind::Subtitle, "eng");
  ass.codec = "ass";
  const LanguagePriorities en{.audio = {"jpn"}, .subtitles = {"en"}};
  CHECK(choose_tracks(std::vector<TrackInfo>{track(1, TrackKind::Audio, "jpn"), pgs, ass}, en).subtitle == 3);
  CHECK_FALSE(
      choose_tracks(std::vector<TrackInfo>{track(1, TrackKind::Audio, "jpn"), pgs}, en).subtitle.has_value());
}

TEST_CASE("selection : deux préférences audio + sous-titres, la première entière gagne",
          "[media][selection][presets]") {
  using ajp::media::LanguagePreset;
  const LanguagePriorities jp_fr_then_fr{
      .presets = {LanguagePreset{"jpn", "fre"}, LanguagePreset{"fre", ""}}};
  const std::vector<TrackInfo> both{track(1, TrackKind::Audio, "fre"), track(2, TrackKind::Audio, "jpn"),
                                    track(3, TrackKind::Subtitle, "fre"),
                                    track(4, TrackKind::Subtitle, "eng")};
  const auto first = choose_tracks(both, jp_fr_then_fr);
  CHECK(first.audio == 2);
  CHECK(first.subtitle == 3);

  // Sans sous-titres français : la préférence 1 n'est pas entière, la 2 prend audio français, sans
  // sous-titres.
  const std::vector<TrackInfo> no_subs{track(1, TrackKind::Audio, "fre"), track(2, TrackKind::Audio, "jpn"),
                                       track(4, TrackKind::Subtitle, "eng", true)};
  const auto second = choose_tracks(no_subs, jp_fr_then_fr);
  CHECK(second.audio == 1);
  CHECK_FALSE(second.subtitle.has_value());

  // Sans audio japonais : idem.
  const std::vector<TrackInfo> no_jp{track(1, TrackKind::Audio, "fre"), track(3, TrackKind::Subtitle, "fre")};
  const auto no_japanese = choose_tracks(no_jp, jp_fr_then_fr);
  CHECK(no_japanese.audio == 1);
  CHECK_FALSE(no_japanese.subtitle.has_value());

  // Aucune préférence entière : on retombe sur la meilleure langue de chaque liste.
  const std::vector<TrackInfo> neither{track(1, TrackKind::Audio, "eng", true),
                                       track(2, TrackKind::Audio, "ger")};
  CHECK(choose_tracks(neither, jp_fr_then_fr).audio == 1);
}
