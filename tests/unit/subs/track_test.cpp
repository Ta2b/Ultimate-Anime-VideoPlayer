#include <string>

#include <ass/ass.h>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "subs/ass_library.h"
#include "subs/srt.h"
#include "subs/subtitle_track.h"
#include "support.h"

using kirari::RefusalCode;
using kirari::subs::AssLibrary;
using kirari::subs::SubtitlePacket;
using kirari::subs::SubtitleTrack;
using kirari::subs::TrackKind;

namespace {

// En-tête Matroska (CodecPrivate) : le script de référence sans ses événements.
std::string codec_private() {
  const std::string script(kirari::subs::test::reference_script());
  return script.substr(0, script.find("Dialogue:"));
}

}  // namespace

TEST_CASE("SubtitleTrack : en-tête ASS puis paquets Matroska, doublons écartés", "[track]") {
  auto library = AssLibrary::create();
  REQUIRE(library);
  auto track = SubtitleTrack::from_header(*library, codec_private());
  REQUIRE(track);
  CHECK(track->kind() == TrackKind::Ass);
  CHECK(track->event_count() == 0);

  const SubtitlePacket first{
      .start_us = 1'000'000, .duration_us = 2'000'000, .data = "0,0,Default,,0,0,0,,Un"};
  const SubtitlePacket second{
      .start_us = 4'000'000, .duration_us = 1'000'000, .data = "1,0,Top,,0,0,0,,Deux"};
  REQUIRE(track->add_packet(first));
  REQUIRE(track->add_packet(second));
  REQUIRE(track->add_packet(first));  // revu après un seek : même ReadOrder
  CHECK(track->event_count() == 2);
}

TEST_CASE("SubtitleTrack : refuse un en-tête vide ou sans section ASS", "[track]") {
  auto library = AssLibrary::create();
  REQUIRE(library);
  CHECK(SubtitleTrack::from_header(*library, "").error().code == RefusalCode::SubtitleFormat);
  CHECK(SubtitleTrack::from_header(*library, "pas un script").error().code == RefusalCode::SubtitleFormat);
  CHECK(SubtitleTrack::from_ass(*library, "pas un script").error().code == RefusalCode::SubtitleFormat);
}

TEST_CASE("SubtitleTrack : paquets texte convertis, doublons écartés", "[track]") {
  auto library = AssLibrary::create();
  REQUIRE(library);
  auto track = SubtitleTrack::text(*library);
  REQUIRE(track);
  CHECK(track->kind() == TrackKind::Text);
  const SubtitlePacket line{.start_us = 0, .duration_us = 1'000'000, .data = "<i>Bonjour</i>"};
  REQUIRE(track->add_packet(line));
  REQUIRE(track->add_packet(line));
  REQUIRE(track->add_packet({.start_us = 2'000'000, .duration_us = 1'000'000, .data = "<i>Bonjour</i>"}));
  CHECK(track->event_count() == 2);
}

TEST_CASE("SubtitleTrack : refuse un paquet hors de ses bornes", "[track]") {
  auto library = AssLibrary::create();
  REQUIRE(library);
  auto track = SubtitleTrack::text(*library);
  REQUIRE(track);
  CHECK(track->add_packet({.start_us = 0, .duration_us = -1, .data = "a"}).error().code ==
        RefusalCode::FieldOutOfRange);
  CHECK(track->add_packet({.start_us = 0, .duration_us = kirari::subs::kMaxEventDurationUs + 1, .data = "a"})
            .error()
            .code == RefusalCode::FieldOutOfRange);
  CHECK(track->add_packet({.start_us = kirari::subs::kMaxTimestampUs + 1, .duration_us = 0, .data = "a"})
            .error()
            .code == RefusalCode::FieldOutOfRange);
  CHECK(
      track
          ->add_packet(
              {.start_us = 0, .duration_us = 0, .data = std::string(kirari::subs::kMaxPacketBytes + 1, 'a')})
          .error()
          .code == RefusalCode::FieldOutOfRange);
  CHECK(track->event_count() == 0);
}

TEST_CASE("SubtitleTrack::load : fichiers .ass et .srt externes", "[track]") {
  auto library = AssLibrary::create();
  REQUIRE(library);
  const kirari::subs::test::TempFile ass("piste.ASS", kirari::subs::test::reference_script());
  auto from_ass = SubtitleTrack::load(*library, ass.path());
  REQUIRE(from_ass);
  CHECK(from_ass->event_count() == 5);

  const kirari::subs::test::TempFile srt("piste.srt",
                                         "1\n00:00:01,000 --> 00:00:02,000\nUn\n\n"
                                         "2\n00:00:03,000 --> 00:00:04,000\n<b>Deux</b>\n");
  auto from_srt = SubtitleTrack::load(*library, srt.path());
  REQUIRE(from_srt);
  CHECK(from_srt->kind() == TrackKind::Text);
  CHECK(from_srt->event_count() == 2);
}

TEST_CASE("SubtitleTrack::load : refuse une extension inconnue, un fichier absent ou un SRT invalide",
          "[track]") {
  auto library = AssLibrary::create();
  REQUIRE(library);
  const kirari::subs::test::TempFile vtt("piste.vtt", "WEBVTT\n");
  CHECK(SubtitleTrack::load(*library, vtt.path()).error().code == RefusalCode::SubtitleFormat);
  CHECK(SubtitleTrack::load(*library, "/chemin/absent.ass").error().code == RefusalCode::SubtitleUnreadable);
  const kirari::subs::test::TempFile broken("casse.srt", "1\n00:00:02,000 --> 00:00:01,000\nA\n");
  CHECK(SubtitleTrack::load(*library, broken.path()).error().code == RefusalCode::SubtitleFormat);
}

TEST_CASE("AssLibrary::add_font : refuse une police vide ou sans nom", "[track]") {
  auto library = AssLibrary::create();
  REQUIRE(library);
  const std::vector<std::byte> one{std::byte{1}};
  CHECK(library->add_font("", one).error().code == RefusalCode::FieldOutOfRange);
  CHECK(library->add_font("vide.ttf", {}).error().code == RefusalCode::FieldOutOfRange);
}

TEST_CASE("to_ms : milliseconde la plus proche, demie vers +∞", "[track]") {
  STATIC_REQUIRE(kirari::subs::to_ms(0) == 0);
  STATIC_REQUIRE(kirari::subs::to_ms(499) == 0);
  STATIC_REQUIRE(kirari::subs::to_ms(500) == 1);
  STATIC_REQUIRE(kirari::subs::to_ms(1'234'567) == 1235);
  STATIC_REQUIRE(kirari::subs::to_ms(-499) == 0);
  STATIC_REQUIRE(kirari::subs::to_ms(-500) == 0);
  STATIC_REQUIRE(kirari::subs::to_ms(-501) == -1);
  STATIC_REQUIRE(kirari::subs::to_ms(-1500) == -1);
}

TEST_CASE("SubtitleTrack::from_srt : une cue hors bornes est écartée, pas le fichier (A17-07)", "[track]") {
  auto library = AssLibrary::create();
  REQUIRE(library);
  const std::string srt =
      "1\n00:00:01,000 --> 00:00:03,000\nPremière réplique\n\n"
      "2\n00:00:00,000 --> 30:00:00,000\nsubs by Groupe\n\n"
      "3\n00:00:05,000 --> 00:00:07,000\nTroisième réplique\n";
  auto track = SubtitleTrack::from_srt(*library, srt);
  REQUIRE(track);
  CHECK(track->event_count() == 2);
}

TEST_CASE("SubtitleTrack : plafond cumulé des paquets, dit une fois (A17-14)", "[track]") {
  auto library = AssLibrary::create();
  REQUIRE(library);
  auto track = SubtitleTrack::text(*library);
  REQUIRE(track);
  // Des paquets de 1 Mo, tous distincts : le plafond tombe après kMaxTrackBytes.
  const std::size_t per_packet = kirari::subs::kMaxPacketBytes;
  const std::size_t fill = kirari::subs::kMaxTrackBytes / per_packet;
  for (std::size_t i = 0; i < fill; ++i) {
    const SubtitlePacket packet{.start_us = static_cast<std::int64_t>(i) * 1'000'000,
                                .duration_us = 500'000,
                                .data = std::string(per_packet, 'a')};
    REQUIRE(track->add_packet(packet));
  }
  const int kept = track->event_count();
  CHECK(kept == static_cast<int>(fill));
  const SubtitlePacket more{.start_us = 999'000'000, .duration_us = 500'000, .data = "encore"};
  const auto refused = track->add_packet(more);
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::FieldOutOfRange);
  const SubtitlePacket again{.start_us = 998'000'000, .duration_us = 500'000, .data = "toujours"};
  CHECK(track->add_packet(again));  // déjà dit : écarté sans refus
  CHECK(track->event_count() == kept);
}

TEST_CASE("SubtitleTrack : les doublons ASS revus après un seek ne remplissent pas la piste (A17-14)",
          "[track]") {
  auto library = AssLibrary::create();
  REQUIRE(library);
  auto track = SubtitleTrack::from_header(*library, codec_private());
  REQUIRE(track);
  const SubtitlePacket big{
      .start_us = 1'000'000,
      .duration_us = 1'000'000,
      .data = "0,0,Default,,0,0,0,," + std::string(kirari::subs::kMaxPacketBytes - 64, 'a')};
  for (std::size_t i = 0; i < 2 * kirari::subs::kMaxTrackBytes / kirari::subs::kMaxPacketBytes; ++i)
    REQUIRE(track->add_packet(big));  // même ReadOrder : un seul événement, compté une fois
  CHECK(track->event_count() == 1);
}

TEST_CASE("SubtitleTrack : substitute_font remplace la police des styles qui la nomment", "[track]") {
  auto library = AssLibrary::create();
  REQUIRE(library);
  auto track = SubtitleTrack::from_ass(*library,
                                       "[Script Info]\nScriptType: v4.00+\n\n[V4+ Styles]\n"
                                       "Format: Name, Fontname, Fontsize, ScaleX\n"
                                       "Style: Default,Trebuchet MS,22,100\n"
                                       "Style: Italique,trebuchet ms,22,100\n"
                                       "Style: Sign,Arial,18,100\n");
  REQUIRE(track);
  CHECK(track->substitute_font("Trebuchet MS", "Gandhi Sans", 1.05) == 2);
  const ass_track* raw = track->get();
  REQUIRE(raw->n_styles >= 3);
  int gandhi = 0;
  for (int i = 0; i < raw->n_styles; ++i) {
    const std::string name = raw->styles[i].FontName;
    if (name == "Gandhi Sans") {
      ++gandhi;
      CHECK(raw->styles[i].ScaleX == Catch::Approx(1.05));
    }
    CHECK(name != "Trebuchet MS");
  }
  CHECK(gandhi == 2);
  CHECK(track->substitute_font("Trebuchet MS", "Gandhi Sans", 1.05) == 0);  // plus rien à remplacer
}
