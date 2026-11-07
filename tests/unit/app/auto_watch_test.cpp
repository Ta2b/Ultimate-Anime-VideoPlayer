// Le suivi du palier de l'Auto (auto_watch.h) : journal au changement, annonce à 3 s, mémoire à 6 s, plancher
// figé dit après 8 s puis toutes les 30 s, mode Rapide toutes les 5 s.
#include "app/play/auto_watch.h"

#include <catch2/catch_test_macros.hpp>

using namespace ajp::app::play;

namespace {
constexpr std::int64_t kSecond = 1'000'000;
}

TEST_CASE("AutoWatch : un palier tenu s'annonce à 3 s, se retient à 6 s", "[app][auto_watch]") {
  AutoWatch watch;
  // Le départ : journalisé, ni annoncé ni retenu avant de tenir.
  CHECK(watch.turn({.at_us = 1, .centi = 200, .can_store = true}).log_change);
  CHECK_FALSE(watch.turn({.at_us = 2 * kSecond, .centi = 200, .can_store = true}).notify);
  // Le premier palier tenu ne s'annonce pas (c'est le départ), mais il se retient à 6 s.
  CHECK_FALSE(watch.turn({.at_us = 4 * kSecond, .centi = 200, .can_store = true}).notify);
  const AutoActions held = watch.turn({.at_us = 7 * kSecond, .centi = 200, .can_store = true});
  REQUIRE(held.store);
  CHECK(*held.store == 200);
  CHECK_FALSE(watch.turn({.at_us = 8 * kSecond, .centi = 200, .can_store = true}).store);  // une seule fois
  // Un changement : annoncé une fois tenu 3 s, retenu à 6 s.
  CHECK(watch.turn({.at_us = 10 * kSecond, .centi = 300, .can_store = true}).log_change);
  CHECK_FALSE(watch.turn({.at_us = 12 * kSecond, .centi = 300, .can_store = true}).notify);
  const AutoActions shown = watch.turn({.at_us = 13 * kSecond, .centi = 300, .can_store = true});
  REQUIRE(shown.notify);
  CHECK(*shown.notify == 300);
  CHECK_FALSE(shown.store);
  CHECK(watch.turn({.at_us = 16 * kSecond, .centi = 300, .can_store = true}).store);
  // Sans mémoire pour cette vidéo, rien ne s'écrit.
  AutoWatch no_store;
  no_store.turn({.at_us = 1, .centi = 200});
  CHECK_FALSE(no_store.turn({.at_us = 10 * kSecond, .centi = 200}).store);
}

TEST_CASE("AutoWatch : plancher figé et mode Rapide au journal, sans le noyer", "[app][auto_watch]") {
  AutoWatch watch;
  watch.turn({.at_us = 1, .centi = 100});
  CHECK_FALSE(watch.turn({.at_us = kSecond, .centi = 100, .at_floor = true}).floor_us);
  const AutoActions stuck = watch.turn({.at_us = 9 * kSecond + 1, .centi = 100, .at_floor = true});
  REQUIRE(stuck.floor_us);
  CHECK(*stuck.floor_us == 8 * kSecond + 1);
  CHECK_FALSE(watch.turn({.at_us = 20 * kSecond, .centi = 100, .at_floor = true}).floor_us);
  CHECK(watch.turn({.at_us = 40 * kSecond, .centi = 100, .at_floor = true}).floor_us);
  // Sorti du plancher : le compte repart.
  watch.turn({.at_us = 41 * kSecond, .centi = 100});
  CHECK_FALSE(watch.turn({.at_us = 72 * kSecond, .centi = 100, .at_floor = true}).floor_us);

  AutoWatch fast;
  CHECK(fast.turn({.at_us = 5 * kSecond, .free = true}).log_free);
  CHECK_FALSE(fast.turn({.at_us = 7 * kSecond, .free = true}).log_free);
  CHECK(fast.turn({.at_us = 10 * kSecond, .free = true}).log_free);
}
