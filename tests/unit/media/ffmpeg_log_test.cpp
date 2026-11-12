#include "media/ffmpeg_log.h"

#include <chrono>
#include <format>

#include <catch2/catch_test_macros.hpp>

using ajp::media::ErrorFolder;
using Clock = std::chrono::steady_clock;

// B15-08 : un flux abîmé change un nombre à chaque image ; ce n'est pas une ligne neuve à chaque fois.
TEST_CASE("ErrorFolder : les lignes qui ne diffèrent que par leurs nombres sont une seule ligne",
          "[media][log]") {
  ErrorFolder folder;
  const auto start = Clock::now();
  CHECK(folder.push("Invalid NAL unit size (123 > 45)", start).size() == 1);
  for (int i = 0; i < 100; ++i)
    CHECK(folder
              .push(std::format("Invalid NAL unit size ({} > {})", 100 + i, 40 + i),
                    start + std::chrono::milliseconds(i))
              .empty());
  const auto next = folder.push("Error splitting the input into NAL units.", start + std::chrono::seconds(2));
  REQUIRE(next.size() == 2);
  CHECK(next[0].find("100 fois de plus") != std::string::npos);
}

TEST_CASE("ErrorFolder : au plus cinq lignes neuves par seconde, le reste est compté", "[media][log]") {
  ErrorFolder folder;
  const auto start = Clock::now();
  int written = 0;
  for (int i = 0; i < 40; ++i)
    written += static_cast<int>(folder
                                    .push(std::format("erreur {}", static_cast<char>('a' + i % 26)) +
                                              std::string(static_cast<std::size_t>(i), 'x'),
                                          start + std::chrono::milliseconds(10 * i))
                                    .size());
  CHECK(written == ErrorFolder::kMaxLinesPerSecond);
  const auto after = folder.push("autre", start + std::chrono::seconds(3));
  REQUIRE(after.size() == 2);
  CHECK(after[0].find("omises") != std::string::npos);
}
