// Le livre des coupes (app/play/cut_book.h, chantier 8.7) : les plans de la pré-analyse disent si une paire
// enjambe une coupe, ou rien hors de leur chronologie.
#include "app/play/cut_book.h"

#include <vector>

#include <catch2/catch_test_macros.hpp>

using namespace ajp::app::play;
using ajp::media::Shot;

namespace {

Shot shot(std::int64_t start_ms, std::int64_t end_ms, bool cut_before = true) {
  Shot s;
  s.start_us = start_ms * 1000;
  s.end_us = end_ms * 1000;
  s.cut_before = cut_before;
  return s;
}

}  // namespace

TEST_CASE("coupes : même plan, plan suivant, hors chronologie", "[app][shots]") {
  // Deux plans analysés depuis un saut (le premier sans coupe connue), un trou, puis un plan plus loin.
  const std::vector<Shot> shots{shot(1000, 2000, false), shot(2000, 3000), shot(5000, 6000)};
  CHECK(known_cut(shots, 1100'000, 1141'000) == false);
  CHECK(known_cut(shots, 1980'000, 2020'000) == true);
  CHECK_FALSE(known_cut(shots, 900'000, 1020'000));   // a avant la chronologie
  CHECK_FALSE(known_cut(shots, 2980'000, 3020'000));  // b dans le trou
  CHECK_FALSE(known_cut(shots, 2900'000, 5010'000));  // le plan de b ne touche pas celui de a
  CHECK_FALSE(known_cut(shots, 2020'000, 2020'000));  // pas une paire
}

TEST_CASE("coupes : un départ d'analyse n'est pas une coupe connue", "[app][shots]") {
  const std::vector<Shot> shots{shot(0, 1000), shot(1000, 2000, false)};
  CHECK_FALSE(known_cut(shots, 980'000, 1020'000));
}

TEST_CASE("coupes : le livre vide ne dit rien, rempli il répond", "[app][shots]") {
  CutBook book;
  CHECK_FALSE(book.active());
  CHECK_FALSE(book.cut_between(0, 40'000));
  const std::vector<Shot> shots{shot(0, 1000), shot(1000, 2000)};
  book.set_shots(shots);
  CHECK(book.active());
  CHECK(book.cut_between(980'000, 1020'000) == true);
  CHECK(book.cut_between(100'000, 140'000) == false);
}
