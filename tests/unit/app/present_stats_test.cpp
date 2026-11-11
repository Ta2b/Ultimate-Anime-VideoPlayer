#include "app/play/present_stats.h"

#include <algorithm>
#include <sstream>
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

TEST_CASE("B32-04 : --stats s'écrit sur le fil d'écriture, toutes les lignes, dans l'ordre", "[app][stats]") {
  std::ostringstream out;
  {
    LatestWriter writer;
    {
      StatsSink sink(out, writer);
      for (std::int64_t seq = 1; seq <= 2000; ++seq)
        sink.add(PresentRecord{.seq = seq, .host_us = seq}, seq * 1000);  // deux secondes : quatre paquets
      writer.flush();
      // Le dernier paquet attend encore la demi-seconde : le puits ne l'a pas posté.
      CHECK(std::count(out.view().begin(), out.view().end(), '\n') < 2000);
    }  // le puits poste le reste en partant
    writer.flush();
  }
  const std::string text = out.str();
  CHECK(std::count(text.begin(), text.end(), '\n') == 2000);
  std::istringstream lines(text);
  std::string line;
  std::int64_t expected = 1;
  bool ordered = true;
  while (std::getline(lines, line)) ordered = ordered && line.starts_with(std::to_string(expected++) + "\t");
  CHECK(ordered);
}

}  // namespace ajp::app::play
