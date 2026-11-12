// --stats : une ligne par présentation. L'en-tête et la ligne viennent du même tableau de colonnes, pour
// qu'ils ne se désaccordent plus (l'en-tête d'avant n'en nommait que 9 sur 10).
#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>

#include "app/play/latest_writer.h"
#include "app/play/session.h"

namespace ajp::app::play {

inline constexpr int kStatsColumns = 20;

// Les noms des colonnes, séparés par des tabulations, terminés par un saut de ligne.
std::string stats_header();
// La ligne d'une présentation, dans le même ordre, terminée par un saut de ligne.
std::string stats_line(const PresentRecord& record);

// B32-04 : les lignes de --stats écrites par le fil d'écriture (LatestWriter), par paquets d'une demi-seconde
// au plus, dans l'ordre ; le thread de présentation ne fait plus qu'ajouter une ligne à un texte. out et
// writer vivent plus longtemps que le puits ; out n'est écrit que par writer.
class StatsSink {
 public:
  StatsSink(std::ostream& out, LatestWriter& writer) : out_(out), writer_(writer) {}
  ~StatsSink() { flush(); }
  StatsSink(const StatsSink&) = delete;
  StatsSink& operator=(const StatsSink&) = delete;

  void add(const PresentRecord& record, std::int64_t now_us);
  // Le reste, posté tout de suite.
  void flush();

 private:
  static constexpr std::int64_t kEveryUs = 500'000;
  static constexpr std::size_t kMaxBytes = 64U << 10U;
  std::ostream& out_;
  LatestWriter& writer_;
  std::string pending_;
  std::uint64_t chunks_ = 0;
  std::int64_t posted_at_us_ = 0;
};

}  // namespace ajp::app::play
