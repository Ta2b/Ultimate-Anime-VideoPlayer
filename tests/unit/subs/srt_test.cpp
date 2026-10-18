#include "subs/srt.h"

#include <catch2/catch_test_macros.hpp>

using ajp::RefusalCode;
using ajp::subs::parse_srt;
using ajp::subs::srt_text_to_ass;

TEST_CASE("parse_srt : numéros, CRLF, BOM et horodatages exacts", "[srt]") {
  const auto cues = parse_srt(
      "\xEF\xBB\xBF"
      "1\r\n"
      "00:00:01,500 --> 00:00:03,250\r\n"
      "Première ligne\r\n"
      "<i>seconde</i>\r\n"
      "\r\n"
      "2\r\n"
      "01:02:03.004 --> 01:02:04.000 X1:10 X2:20\r\n"
      "Fin\r\n");
  REQUIRE(cues);
  REQUIRE(cues->size() == 2);
  CHECK((*cues)[0].start_us == 1'500'000);
  CHECK((*cues)[0].end_us == 3'250'000);
  CHECK((*cues)[0].text == "Première ligne\n<i>seconde</i>");
  CHECK((*cues)[1].start_us == ((1 * 60 + 2) * 60 + 3) * 1'000'000LL + 4'000);
  CHECK((*cues)[1].end_us == ((1 * 60 + 2) * 60 + 4) * 1'000'000LL);
  CHECK((*cues)[1].text == "Fin");
}

TEST_CASE("parse_srt : numéros facultatifs, lignes vides en trop, fichier vide", "[srt]") {
  const auto cues =
      parse_srt("\n\n00:00:00,000 --> 00:00:01,000\nA\n\n\n\n00:00:02,000 --> 00:00:03,000\nB\n");
  REQUIRE(cues);
  REQUIRE(cues->size() == 2);
  CHECK((*cues)[1].text == "B");
  const auto empty = parse_srt("");
  REQUIRE(empty);
  CHECK(empty->empty());
}

TEST_CASE("parse_srt : les horodatages délimitent les cues, même sans ligne vide", "[srt]") {
  // Numéro collé au texte précédent, ligne vide dans un texte, texte numérique :
  // le démultiplexeur SRT de FFmpeg (lecteur actuel) lit ces fichiers ainsi.
  const auto cues = parse_srt(
      "1\n00:00:01,000 --> 00:00:02,000\nA\n"
      "2\n00:00:03,000 --> 00:00:04,000\nB1\n\nB2\n\n"
      "3\n00:00:05,000 --> 00:00:06,000\n42\n2009\n");
  REQUIRE(cues);
  REQUIRE(cues->size() == 3);
  CHECK((*cues)[0].text == "A");
  CHECK((*cues)[1].start_us == 3'000'000);
  CHECK((*cues)[1].text == "B1\nB2");
  CHECK((*cues)[2].text == "42\n2009");  // un nombre sans horodatage après lui reste du texte

  const auto bare = parse_srt("00:00:01,000 --> 00:00:02,000\nA\n00:00:03,000 --> 00:00:04,000\nB\n");
  REQUIRE(bare);
  REQUIRE(bare->size() == 2);
  CHECK((*bare)[0].text == "A");
  CHECK((*bare)[1].text == "B");
}

TEST_CASE("parse_srt : un horodatage mal formé au milieu du fichier est refusé à sa ligne", "[srt]") {
  const auto broken = parse_srt("1\n00:00:01,000 --> 00:00:02,000\nA\n\n2\n00:00:03,000 --> 00:00:04\nB\n");
  REQUIRE_FALSE(broken);
  CHECK(broken.error().code == RefusalCode::SubtitleFormat);
  CHECK(broken.error().detail.find("ligne 6") != std::string::npos);
  // Une flèche dans un dialogue n'est pas un horodatage.
  const auto arrow = parse_srt("00:00:01,000 --> 00:00:02,000\nA --> B\n");
  REQUIRE(arrow);
  CHECK(arrow->front().text == "A --> B");
}

TEST_CASE("parse_srt : refuse un horodatage hors plage ou mal formé, avec la ligne", "[srt]") {
  const auto minutes = parse_srt("1\n00:61:00,000 --> 00:62:00,000\nA\n");
  REQUIRE_FALSE(minutes);
  CHECK(minutes.error().code == RefusalCode::SubtitleFormat);
  CHECK(minutes.error().detail.find("ligne 2") != std::string::npos);

  const auto backwards = parse_srt("00:00:05,000 --> 00:00:04,000\nA\n");
  REQUIRE_FALSE(backwards);
  CHECK(backwards.error().detail.find("fin avant début") != std::string::npos);

  CHECK_FALSE(parse_srt("00:00:01,5 --> 00:00:02,000\nA\n"));        // millisecondes sur 3 chiffres
  CHECK_FALSE(parse_srt("00:00:01,000 -> 00:00:02,000\nA\n"));       // flèche
  CHECK_FALSE(parse_srt("00:00:01,000 --> 00:00:02,000junk\nA\n"));  // suite collée
  CHECK_FALSE(parse_srt("Bonjour\n"));
  CHECK_FALSE(parse_srt("1\n"));
}

TEST_CASE("srt_text_to_ass : styles, retours à la ligne et blocs ASS", "[srt]") {
  CHECK(srt_text_to_ass("<i>a</i> <B>b</B> <u>c</u> <s>d</s>") ==
        "{\\i1}a{\\i0} {\\b1}b{\\b0} {\\u1}c{\\u0} {\\s1}d{\\s0}");
  CHECK(srt_text_to_ass("un\ndeux\r\ntrois") == "un\\Ndeux\\Ntrois");
  CHECK(srt_text_to_ass("{\\an8}en haut") == "{\\an8}en haut");
  CHECK(srt_text_to_ass("je <3 les sous-titres") == "je <3 les sous-titres");
  CHECK(srt_text_to_ass("a < b > c") == "a < b > c");
  CHECK(srt_text_to_ass("<i>pas fermée") == "{\\i1}pas fermée");
}

TEST_CASE("srt_text_to_ass : couleurs et polices, restaurées à la fermeture", "[srt]") {
  CHECK(srt_text_to_ass("<font color=\"#FF8000\">orange</font>") == "{\\c&H0080FF&}orange{\\c}");
  CHECK(srt_text_to_ass("<font color=red>rouge</font>") == "{\\c&H0000FF&}rouge{\\c}");
  CHECK(srt_text_to_ass("<font face='Noto Sans' size=\"20\">x</font>") == "{\\fnNoto Sans\\fs20}x{\\fn\\fs}");
  // Imbrication : la fermeture rend la couleur de la balise englobante.
  CHECK(srt_text_to_ass("<font color=\"#0000FF\">b<font color=\"#00FF00\">v</font>b</font>") ==
        "{\\c&HFF0000&}b{\\c&H00FF00&}v{\\c&HFF0000&}b{\\c}");
  // Couleur illisible : la balise est reconnue, sans effet.
  CHECK(srt_text_to_ass("<font color=\"bleu-roi\">x</font>") == "x");
}

TEST_CASE("text_track_header : un style Default et un format d'événements", "[srt]") {
  const std::string_view header = ajp::subs::text_track_header();
  CHECK(header.find("[Script Info]") != std::string_view::npos);
  CHECK(header.find("Style: Default,") != std::string_view::npos);
  CHECK(header.find("Format: Layer, Start, End, Style") != std::string_view::npos);
}
