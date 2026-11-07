// Les horodatages de av_util.h aux bornes (P2.29) : ni soustraction indéfinie ni faux « sans horodatage ».
#include "media/av_util.h"

#include <cstdint>
#include <limits>

#include <catch2/catch_test_macros.hpp>

#include "core/frame.h"

TEST_CASE("av_util : un horodatage qui déborde reste sans horodatage, le décalage sature", "[media]") {
  constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
  // 1 s par tic : av_rescale_q déborde et rend INT64_MIN.
  CHECK(ajp::media::to_us(kMax / 2, AVRational{1, 1}, 0) == ajp::kNoPts);
  CHECK(ajp::media::to_us(-kMax / 2, AVRational{1, 1}, 0) == ajp::kNoPts);
  CHECK(ajp::media::to_us(-1'000, AVRational{1, 1'000'000}, kMax) == ajp::kNoPts + 1);
  CHECK(ajp::media::to_us(2, AVRational{1, 1}, 500'000) == 1'500'000);
  CHECK(ajp::media::from_us(kMax, AVRational{1, 1'000'000}, 10) == kMax);
}
