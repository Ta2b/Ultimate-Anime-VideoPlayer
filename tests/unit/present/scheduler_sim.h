// Simulation d'une lecture : gpu remplit un ring de slots à partir des instants de la cadence
// (CadenceAccumulator), l'ordonnanceur choisit une image par vsync et rend les slots. Le
// simulateur vérifie le protocole du ring à chaque rendu et note quelle image passe à chaque
// vsync.
#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "core/cadence.h"
#include "core/clock.h"
#include "core/ring.h"
#include "present/scheduler.h"

namespace ajp::present::test {

// 23,976 im/s : une source toutes les 1001/24 ms.
inline constexpr double kNtscFilmFps = 24000.0 / 1001.0;

struct SimOptions {
  double source_fps = kNtscFilmFps;
  core::Ratio factor{1, 1};  // facteur d'interpolation ; 1 : la source telle quelle
  int ring_slots = 4;
  double period_us = 0.0;  // période réelle des vsyncs
  std::uint32_t generation = 7;
};

// Ce qui s'est passé à une vsync.
struct VsyncRecord {
  std::int64_t host_us = 0;
  std::optional<std::int64_t> frame;  // rang de l'image de sortie montrée
  bool changed = false;
  ClockSnapshot clock;
};

class Simulation {
 public:
  Simulation(const SchedulerConfig& config, const SimOptions& options);

  PresentScheduler& scheduler() { return scheduler_; }

  // gpu ne produit pas l'image de sortie j avant la vsync rendue par ce rappel (en ordre :
  // une image retenue retient les suivantes).
  void set_delay(std::function<std::optional<std::int64_t>(std::int64_t frame)> delay) {
    delay_ = std::move(delay);
  }
  // Horloge maître donnée à chaque vsync (Vrr) ; invalide par défaut.
  void set_master(std::function<ClockSnapshot(std::int64_t host_us)> master) { master_ = std::move(master); }

  // Une vsync de rang index à host_us : gpu produit ce qu'il peut, puis l'ordonnanceur décide.
  const VsyncRecord& tick(std::int64_t index, std::int64_t host_us);
  // n vsyncs régulières à la période de SimOptions.
  void run(int vsyncs);

  // Seek : époque suivante, et la sortie repart de la source source_index.
  void seek(std::uint32_t epoch, std::int64_t source_index);

  const std::vector<VsyncRecord>& records() const { return records_; }
  std::int64_t produced() const { return produced_; }
  std::int64_t pts_of(std::int64_t frame) const { return pts_.at(static_cast<std::size_t>(frame)); }
  std::uint32_t epoch_of(std::int64_t frame) const { return epochs_.at(static_cast<std::size_t>(frame)); }
  int releases_displayed() const { return releases_displayed_; }
  int releases_unseen() const { return releases_unseen_; }
  int busy_slots() const;

  // Nombre de vsyncs consécutives de chaque image montrée, dans l'ordre, entre deux rangs.
  std::vector<std::pair<std::int64_t, int>> runs(std::size_t from, std::size_t to) const;

 private:
  struct Instant {
    std::int64_t pts = kNoPts;
    SourceId left;
    SourceId right;
    core::Ratio t;
  };
  Instant next_instant();
  void produce(std::int64_t index);

  PresentScheduler scheduler_;
  core::CadenceAccumulator cadence_;
  SimOptions options_;
  std::deque<Instant> instants_;
  std::int64_t next_source_ = 0;
  std::uint32_t epoch_ = 0;
  std::int64_t produced_ = 0;
  std::int64_t next_index_ = 0;
  std::vector<bool> busy_;
  std::vector<std::uint64_t> released_value_;  // dernière valeur rendue de chaque slot
  std::map<std::pair<std::uint32_t, std::uint64_t>, std::int64_t> frame_of_;  // (slot, valeur) -> rang
  std::vector<std::int64_t> pts_;
  std::vector<std::uint32_t> epochs_;
  std::vector<VsyncRecord> records_;
  std::function<std::optional<std::int64_t>(std::int64_t)> delay_;
  std::function<ClockSnapshot(std::int64_t)> master_;
  int releases_displayed_ = 0;
  int releases_unseen_ = 0;
};

// Durée d'une source en microsecondes, et l'horodatage de la source i.
std::int64_t source_pts(double fps, std::int64_t index);

}  // namespace ajp::present::test
