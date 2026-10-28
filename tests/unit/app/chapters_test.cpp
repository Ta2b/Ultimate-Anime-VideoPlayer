// Chapitres : chapitre courant, pas vers le suivant et le précédent, libellé du message bref.
#include "app/play/chapters.h"

#include <vector>

#include <catch2/catch_test_macros.hpp>

using namespace ajp::app::play;

namespace {

const std::vector<ChapterMark> kChapters{
    {.start_us = 0, .title = "Avant-générique"},
    {.start_us = 90'000'000, .title = "Opening"},
    {.start_us = 180'000'000, .title = ""},
    {.start_us = 1'300'000'000, .title = "Ending"},
};

}  // namespace

TEST_CASE("chapitres : le chapitre courant est le dernier commencé", "[app][chapters]") {
  CHECK(chapter_at(kChapters, 0) == 0U);
  CHECK(chapter_at(kChapters, 89'999'999) == 0U);
  CHECK(chapter_at(kChapters, 90'000'000) == 1U);
  CHECK(chapter_at(kChapters, 2'000'000'000) == 3U);
  const std::vector<ChapterMark> late{{.start_us = 5'000'000, .title = "A"}};
  CHECK_FALSE(chapter_at(late, 1'000'000));
  CHECK_FALSE(chapter_at({}, 0));
}

TEST_CASE("chapitres : suivant, puis rien après le dernier", "[app][chapters]") {
  CHECK(chapter_step(kChapters, 10'000'000, 1) == 1U);
  CHECK(chapter_step(kChapters, 90'000'000, 1) == 2U);
  CHECK_FALSE(chapter_step(kChapters, 1'400'000'000, 1));
  const std::vector<ChapterMark> late{{.start_us = 5'000'000, .title = "A"}};
  CHECK(chapter_step(late, 0, 1) == 0U);  // avant le premier chapitre : le premier
  CHECK_FALSE(chapter_step({}, 0, 1));
}

TEST_CASE("chapitres : précédent revient au début du chapitre, puis au chapitre d'avant", "[app][chapters]") {
  // Bien entré dans le chapitre : son début.
  CHECK(chapter_step(kChapters, 120'000'000, -1) == 1U);
  // Juste après son début : le chapitre d'avant.
  CHECK(chapter_step(kChapters, 90'000'000 + kChapterRestartUs, -1) == 0U);
  CHECK(chapter_step(kChapters, 91'000'000, -1) == 0U);
  // Dans le premier : toujours son début.
  CHECK(chapter_step(kChapters, 1'000'000, -1) == 0U);
}

TEST_CASE("chapitres : libellé avec ou sans titre", "[app][chapters]") {
  CHECK(chapter_label(kChapters, 1) == "Chapitre 2/4 · Opening");
  CHECK(chapter_label(kChapters, 2) == "Chapitre 3/4");
}

TEST_CASE("chapitres : les génériques se reconnaissent à leur titre", "[app][chapters]") {
  for (const char* title : {"Opening", "OP", "OP2", "ED 1", "Ending", "Intro", "Générique de fin", "Credits"})
    CHECK(is_credits_chapter({.start_us = 0, .title = title}));
  for (const char* title : {"", "Avant-générique", "Partie A", "Preview", "Episode", "Cooperation"})
    CHECK_FALSE(is_credits_chapter({.start_us = 0, .title = title}));
}

TEST_CASE("chapitres : passer le générique mène au chapitre suivant ou à la fin", "[app][chapters]") {
  constexpr std::int64_t kDuration = 1'440'000'000;
  CHECK(credits_skip_target(kChapters, 100'000'000, kDuration) == 180'000'000);  // dans « Opening »
  CHECK(credits_skip_target(kChapters, 1'350'000'000, kDuration) == kDuration);  // « Ending » est le dernier
  CHECK_FALSE(credits_skip_target(kChapters, 10'000'000, kDuration));            // « Avant-générique »
  CHECK_FALSE(credits_skip_target({}, 10'000'000, kDuration));
}
