#include "audio/pcm.h"

#include <catch2/catch_test_macros.hpp>

#include "audio/timeline.h"

using ajp::RefusalCode;
using namespace ajp::audio;

TEST_CASE("pcm : conversions trames et microsecondes arrondies au plus proche", "[audio][pcm]") {
  CHECK(frames_to_us(48'000, 48'000) == 1'000'000);
  CHECK(frames_to_us(1, 48'000) == 21);    // 20,83 µs
  CHECK(frames_to_us(-1, 48'000) == -21);  // symétrique
  CHECK(frames_to_us(1, 44'100) == 23);    // 22,68 µs
  CHECK(us_to_frames(1'000'000, 44'100) == 44'100);
  CHECK(us_to_frames(11, 48'000) == 1);  // 0,528 trame
  CHECK(us_to_frames(-11, 48'000) == -1);
}

TEST_CASE("pcm : un format hors bornes est refusé, pas corrigé", "[audio][pcm]") {
  CHECK(checked({.sample_rate = 48'000, .channels = 8}).has_value());
  CHECK(checked({.sample_rate = 8'000, .channels = 1}).has_value());
  for (const PcmFormat bad :
       {PcmFormat{.sample_rate = 48'000, .channels = 9}, PcmFormat{.sample_rate = 48'000, .channels = 0},
        PcmFormat{.sample_rate = 7'999, .channels = 2}, PcmFormat{.sample_rate = 400'000, .channels = 2}}) {
    const auto refused = checked(bad);
    REQUIRE_FALSE(refused);
    CHECK(refused.error().code == RefusalCode::FieldOutOfRange);
  }
}

TEST_CASE("pcm : drop_front retire des trames entières et avance le temps", "[audio][pcm]") {
  PcmBlock block{.pts_us = 1'000'000, .channels = 2, .samples = {1, 2, 3, 4, 5, 6, 7, 8}};
  block.drop_front(1, 48'000);
  CHECK(block.frames() == 3);
  CHECK(block.samples.front() == 3.0F);
  CHECK(block.pts_us == 1'000'021);
  block.drop_front(10, 48'000);  // plus que le bloc : tout part
  CHECK(block.frames() == 0);
}

TEST_CASE("Timeline : chaque trame retrouve son temps média", "[audio][timeline]") {
  Timeline timeline(48'000);
  CHECK_FALSE(timeline.pts_at(0));
  timeline.append(0, 4'800, 2'000'000);       // 100 ms à partir de 2 s
  timeline.append(4'800, 480, std::nullopt);  // 10 ms de silence de calage
  timeline.append(5'280, 4'800, 2'100'000);
  CHECK(timeline.pts_at(0) == 2'000'000);
  CHECK(timeline.pts_at(2'400) == 2'050'000);
  CHECK_FALSE(timeline.pts_at(4'800));  // un silence de calage n'a pas de temps média
  CHECK_FALSE(timeline.pts_at(5'000));
  CHECK(timeline.pts_at(5'280) == 2'100'000);
  CHECK(timeline.pts_at(5'280 + 480) == 2'110'000);
  CHECK(timeline.pts_at(20'000) == 2'200'000);  // au-delà de la fin : la fin
}

TEST_CASE("Timeline : un index d'avant le premier segment est inconnu", "[audio][timeline]") {
  Timeline timeline(48'000);
  timeline.append(10'000, 4'800, 5'000'000);
  CHECK_FALSE(timeline.pts_at(9'999));  // mesure périmée, d'avant un vidage
  CHECK(timeline.pts_at(10'000) == 5'000'000);
  timeline.append(14'800, 4'800, 5'100'000);
  timeline.forget_before(16'000);
  CHECK_FALSE(timeline.pts_at(10'000));
  CHECK(timeline.pts_at(16'000) == 5'125'000);
  timeline.clear();
  CHECK(timeline.empty());
}
