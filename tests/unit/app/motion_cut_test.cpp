// « Couper l'interpolation si la carte ne suit pas » (app/play/motion_cut.h) : le seuil d'une image
// interpolée par paire, la dizaine de secondes de suite, ce qui ne compte pas (seek, pause, plan presque
// fixe), et l'option dans play.toml, lisible dans un ancien fichier.
#include "app/play/motion_cut.h"

#include <cstdint>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "app/play/settings.h"

namespace ajp::app::play {
namespace {

// La boucle du lecteur relit l'écran toutes les 500 ms ; une source à 24 i/s donne 12 paires par relevé.
constexpr std::int64_t kPollUs = 500'000;

struct Feed {
  MotionWatch watch;
  MotionTally tally;
  std::int64_t now_us = 0;

  // `seconds` secondes de lecture : `owed` paires dues et `made` images interpolées par relevé. Vrai si la
  // coupure tombe.
  bool play(double seconds, std::uint64_t owed, std::uint64_t made, bool counting = true) {
    bool cut = false;
    for (int i = 0; i < static_cast<int>(seconds * 2.0); ++i) {
      now_us += kPollUs;
      tally.owed += owed;
      tally.made += made;
      cut = watch.note(now_us, tally, counting) || cut;
    }
    return cut;
  }
};

}  // namespace

TEST_CASE("motion cut : RIFE qui tient une image par paire n'est jamais coupé", "[app][motion_cut]") {
  Feed feed;
  CHECK_FALSE(feed.play(120.0, 12, 12));  // juste une image par paire : il tient
  CHECK_FALSE(feed.play(120.0, 12, 30));
  CHECK(feed.watch.behind() == 0);
}

TEST_CASE("motion cut : moins d'une image par paire pendant 10 s coupe, pas avant", "[app][motion_cut]") {
  Feed feed;
  CHECK_FALSE(feed.play(5.0, 12, 30));  // la mise en route
  // RIFE en retard : l'image d'origine reste à l'écran, quelques paires interpolées encore.
  CHECK_FALSE(feed.play(8.0, 12, 6));
  CHECK(feed.watch.behind() >= 7);
  CHECK(feed.play(3.0, 12, 6));
  CHECK(feed.watch.behind() >= MotionWatch::kCutWindows);
}

TEST_CASE("motion cut : une seconde qui tient remet le compte à zéro", "[app][motion_cut]") {
  Feed feed;
  CHECK_FALSE(feed.play(3.0, 12, 30));
  CHECK_FALSE(feed.play(8.0, 12, 0));
  CHECK_FALSE(feed.play(1.0, 12, 24));  // RIFE remonte : les images interpolées reviennent
  CHECK(feed.watch.behind() == 0);
  CHECK_FALSE(feed.play(8.0, 12, 0));  // de nouveau en retard : le compte repart
  CHECK(feed.play(4.0, 12, 0));
}

TEST_CASE("motion cut : seeks, pauses et plans presque fixes ne comptent pas", "[app][motion_cut]") {
  Feed feed;
  CHECK_FALSE(feed.play(3.0, 12, 30));
  CHECK_FALSE(feed.play(8.0, 12, 0));
  const int before = feed.watch.behind();
  // Une pause : rien ne se compte, ni retard ni rattrapage ; puis la reprise attend kSettleUs.
  CHECK_FALSE(feed.play(60.0, 0, 0, false));
  CHECK_FALSE(feed.play(2.0, 12, 0));  // la file se remplit après la reprise
  CHECK(feed.watch.behind() == before);
  // Un seek : de même.
  ++feed.tally.restarts;
  CHECK_FALSE(feed.play(2.0, 12, 0));
  CHECK(feed.watch.behind() == before);
  // Un dessin tenu longtemps : une paire due par seconde, trop peu pour juger.
  CHECK_FALSE(feed.play(30.0, 0, 0));
  CHECK(feed.watch.behind() == before);
  // Le retard reprend là où il en était : la coupure tombe sans attendre 10 s de plus.
  CHECK(feed.play(static_cast<double>(MotionWatch::kCutWindows - before), 12, 0));
}

TEST_CASE("motion cut : le choix d'interpolation qui garde la coupure", "[app][motion_cut]") {
  PlaySettings s;
  const InterpolationChoice cut = interpolation_choice(s);
  PlaySettings changed = s;
  changed.volume = 0.2F;  // un réglage hors de l'interpolation ne la rend pas
  CHECK(interpolation_choice(changed) == cut);
  changed.auto_off = false;  // décochée : RIFE continue quoi qu'il arrive
  CHECK_FALSE(interpolation_choice(changed) == cut);
  changed = s;
  changed.auto_free = false;  // Stable choisi à la main
  CHECK_FALSE(interpolation_choice(changed) == cut);
}

// L'option se garde dans play.toml ; un fichier d'avant elle se lit, cochée (le défaut).
TEST_CASE("play.toml : « Couper l'interpolation si la carte ne suit pas », absente d'un ancien fichier",
          "[app][settings][motion_cut]") {
  CHECK(PlaySettings{}.auto_off);
  const std::string old =
      "version = 1\n[interpolation]\nauto = true\nauto_mode = \"libre\"\nauto_cap = 0\nrife = 2\n";
  const auto read = parse_settings(old);
  REQUIRE(read);
  CHECK(read->auto_off);
  CHECK(read->auto_free);
  PlaySettings off;
  off.auto_off = false;
  const std::string text = format_settings(off);
  CHECK(text.find("auto_off = false") != std::string::npos);
  const auto reread = parse_settings(text);
  REQUIRE(reread);
  CHECK_FALSE(reread->auto_off);
  // Elle se change en direct, sans rouvrir la lecture.
  PlaySettings running;
  CHECK_FALSE(live_part(running, off, true).auto_off);
}

}  // namespace ajp::app::play
