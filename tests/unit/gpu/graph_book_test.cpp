#include "gpu/graph_book.h"

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

}  // namespace ajp::gpu
