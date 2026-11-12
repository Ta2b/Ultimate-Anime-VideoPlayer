#include "gpu/graph_book.h"

#include <cstdint>
#include <string>

#include <catch2/catch_test_macros.hpp>

namespace ajp::gpu {

TEST_CASE("Graphes : chaud, capture, rejeu, puis dépassement compté et signalé", "[gpu][graphs]") {
  GraphBook<int> book(16, 3);
  CHECK(book.next(0).step == GraphStep::Warm);
  CHECK(book.next(0).step == GraphStep::Capture);
  book.captured(true);
  CHECK(book.next(0).step == GraphStep::Replay);
  for (int key = 1; key < 16; ++key) CHECK(book.next(key).step == GraphStep::Warm);
  CHECK(book.next(16).step == GraphStep::Overflow);  // 17e clé
  CHECK(book.counters().overflow == 1);
  CHECK_FALSE(book.take_notice().empty());
  CHECK(book.take_notice().empty());  // dit une fois
  CHECK(book.counters().replays == 1);
  CHECK(book.counters().captures == 1);
}

TEST_CASE("Graphes : trois captures refusées coupent les graphes", "[gpu][graphs]") {
  GraphBook<int> book(4, 3);
  for (int attempt = 0; attempt < 3; ++attempt) {
    static_cast<void>(book.next(7));
    if (attempt == 0) continue;  // le premier lancement est chaud
    book.captured(false, "refusée");
  }
  CHECK_FALSE(book.off());  // deux refus seulement
  static_cast<void>(book.next(7));
  book.captured(false, "refusée");
  CHECK(book.off());
  CHECK(book.next(7).step == GraphStep::Direct);
  CHECK(book.take_notice().find("refusée") != std::string::npos);
}

TEST_CASE("Graphes : un rejeu refusé repasse en direct, puis se recapture", "[gpu][graphs]") {
  GraphBook<int> book(4, 3);
  static_cast<void>(book.next(1));  // chaud
  const auto capture = book.next(1);
  REQUIRE(capture.step == GraphStep::Capture);
  book.captured(true);
  const auto replay = book.next(1);
  REQUIRE(replay.step == GraphStep::Replay);
  book.launch_failed(replay.index);
  CHECK(book.counters().launch_failures == 1);
  CHECK_FALSE(book.off());
  CHECK(book.next(1).step == GraphStep::Capture);  // recapturé
  book.captured(true);
  CHECK(book.next(1).step == GraphStep::Replay);
  book.launch_failed(0);
  book.launch_failed(0);
  CHECK(book.off());  // trois échecs : graphes coupés
  CHECK(book.next(1).step == GraphStep::Direct);
  CHECK(book.take_notice().find("rejeu refusé") != std::string::npos);
}

TEST_CASE("Graphes : des refus espacés par assez de rejeux ne coupent plus les graphes (A21-08)",
          "[gpu][graphs]") {
  GraphBook<int> book(4, 3);
  static_cast<void>(book.next(1));  // chaud
  REQUIRE(book.next(1).step == GraphStep::Capture);
  book.captured(true);
  for (int failure = 0; failure < 5; ++failure) {
    const auto replay = book.next(1);
    REQUIRE(replay.step == GraphStep::Replay);
    book.launch_failed(replay.index);
    REQUIRE(book.next(1).step == GraphStep::Capture);
    book.captured(true);
    for (std::uint64_t i = 0; i < GraphBook<int>::kHealAfterReplays; ++i) {
      REQUIRE(book.next(1).step == GraphStep::Replay);
    }
  }
  CHECK_FALSE(book.off());
  CHECK(book.counters().launch_failures == 5);  // le total reste au journal
}

TEST_CASE("Graphes : clear remet à zéro les séries d'échecs et rouvre les graphes (A21-08)",
          "[gpu][graphs]") {
  GraphBook<int> book(4, 3);
  for (int attempt = 0; attempt < 4; ++attempt) {
    static_cast<void>(book.next(7));
    if (attempt > 0) book.captured(false, "refusée");
  }
  REQUIRE(book.off());
  book.clear();  // les adresses ont changé
  CHECK_FALSE(book.off());
  CHECK(book.next(7).step == GraphStep::Warm);
  CHECK(book.counters().capture_failures == 3);
}

TEST_CASE("Graphes : coupés, ils se réessaient après kRetryAfterDirect lancements directs (C21-07)",
          "[gpu][graphs]") {
  GraphBook<int> book(4, 3);
  for (int attempt = 0; attempt < 4; ++attempt) {
    static_cast<void>(book.next(7));
    if (attempt > 0) book.captured(false, "mémoire basse");
  }
  REQUIRE(book.off());
  static_cast<void>(book.take_notice());
  // Le moteur demande skip() avant chaque lancement ; next() compte aussi un lancement direct.
  CHECK(book.next(7).step == GraphStep::Direct);
  for (std::uint64_t i = 2; i < GraphBook<int>::kRetryAfterDirect; ++i) CHECK(book.skip());
  CHECK_FALSE(book.skip());  // le 2000e : les graphes se rouvrent
  CHECK_FALSE(book.off());
  CHECK(book.take_notice().find("réessayés") != std::string::npos);
  // L'entrée gardée en est à sa capture : elle se refait, et une réussite rend les rejeux.
  CHECK(book.next(7).step == GraphStep::Capture);
  book.captured(true);
  CHECK(book.next(7).step == GraphStep::Replay);
  // Trois nouveaux refus de suite les recoupent pour un autre tour.
  CHECK(book.next(8).step == GraphStep::Warm);
  for (int attempt = 0; attempt < 3; ++attempt) {
    CHECK(book.next(8).step == GraphStep::Capture);
    book.captured(false, "mémoire basse");
  }
  CHECK(book.off());
  CHECK(book.skip());
}

TEST_CASE("Graphes : registre plein, une entrée qui ne sert plus laisse sa place (A21-08)", "[gpu][graphs]") {
  GraphBook<int> book(2, 3);
  static_cast<void>(book.next(1));
  static_cast<void>(book.next(2));
  CHECK(book.next(3).step == GraphStep::Overflow);  // 1 et 2 viennent de servir
  // La clé 1 reste vivante, la clé 2 ne revient plus (ancienne génération de l'anneau).
  for (std::uint64_t i = 0; i <= GraphBook<int>::kStaleAfter * 2; ++i) static_cast<void>(book.next(1));
  const auto taken = book.next(3);
  CHECK(taken.step == GraphStep::Warm);
  CHECK(taken.index == 1);  // la place de la clé 2
  CHECK(book.size() == 2);
  CHECK(book.next(1).step != GraphStep::Overflow);  // la clé vivante n'a pas été chassée
}

TEST_CASE("Graphes : des refus de capture isolés, chacun suivi d'une réussite, ne coupent rien (B21-02)",
          "[gpu][graphs]") {
  GraphBook<int> book(8, 3);
  for (int key = 0; key < 3; ++key) {
    static_cast<void>(book.next(key));  // chaud
    REQUIRE(book.next(key).step == GraphStep::Capture);
    book.captured(false, "mémoire juste");
    REQUIRE(book.next(key).step == GraphStep::Capture);  // réessayée
    book.captured(true);
  }
  CHECK_FALSE(book.off());  // trois refus, mais jamais de suite
  CHECK(book.next(2).step == GraphStep::Replay);
  CHECK(book.counters().capture_failures == 3);
  CHECK(book.counters().captures == 3);
}

TEST_CASE("Graphes : un registre de capacité nulle garde une place (B21-03)", "[gpu][graphs]") {
  GraphBook<int> book(0, 3);
  CHECK(book.next(1).step == GraphStep::Warm);
  CHECK(book.next(2).step == GraphStep::Overflow);  // la seule place sert encore à la clé 1
  CHECK(book.size() == 1);
}

}  // namespace ajp::gpu
