#include "app/play/present_stats.h"

#include <algorithm>
#include <string>

#include <catch2/catch_test_macros.hpp>

namespace ajp::app::play {

TEST_CASE("Option --stats : l'en-tête nomme chaque colonne écrite, dans l'ordre", "[app][stats]") {
  const PresentRecord record{.seq = 7, .host_us = 1000, .why = "i", .display_gpu_us = 42};
  const std::string header = stats_header();
  const std::string line = stats_line(record);
  const auto tabs = [](const std::string& text) { return std::count(text.begin(), text.end(), '\t'); };
  CHECK(tabs(header) == kStatsColumns - 1);
  CHECK(tabs(line) == kStatsColumns - 1);
  CHECK(header.starts_with("seq\thost_us\t"));
  CHECK(header.ends_with("display_gpu_us\tpresent_id\tfeedback_id\ttarget_us\tvisible_us\tvsync_k\tlevel\n"));
  CHECK(line.starts_with("7\t1000\t"));
  CHECK(line.ends_with("\t42\t0\t0\t0\t0\t-1\topen\n"));
}

}  // namespace ajp::app::play
