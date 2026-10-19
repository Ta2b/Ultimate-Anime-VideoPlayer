// Progression de l'export : images faites et totales, débit en images par seconde, temps
// restant. Le compteur ne lit pas l'horloge lui-même : on lui donne l'instant de chaque mesure,
// si bien qu'un test le pilote sans attendre.
#pragma once

#include <chrono>
#include <cstdint>
#include <optional>

namespace ajp::exporter {

using ExportClock = std::chrono::steady_clock;

struct ExportProgress {
  std::int64_t frames_done = 0;
  std::optional<std::int64_t> frames_total;  // inconnu quand la source ne le dit pas
  double frames_per_second = 0.0;            // moyen depuis la première image ; 0 avant
  std::chrono::milliseconds elapsed{0};
  std::optional<std::chrono::milliseconds> remaining;  // connu avec le total et un débit non nul
  bool finishing = false;                              // toutes les images sont encodées : mux final en cours
};

class ProgressMeter {
 public:
  explicit ProgressMeter(std::optional<std::int64_t> frames_total) : total_(frames_total) {}

  // Pose l'instant de départ (ouverture faite, première image demandée).
  void start(ExportClock::time_point now) { start_ = now; }

  // Une image de plus, encodée à l'instant now ; rend l'état à publier.
  ExportProgress frame_done(ExportClock::time_point now);

  // L'état courant à l'instant now, sans image de plus.
  ExportProgress at(ExportClock::time_point now) const;

  std::int64_t frames_done() const { return done_; }

 private:
  std::optional<std::int64_t> total_;
  std::optional<ExportClock::time_point> start_;
  std::int64_t done_ = 0;
};

}  // namespace ajp::exporter
