// La pré-analyse des plans : un second décodage du fichier lu, vidéo seule, en tâche de fond et en avance
// sur la lecture. Chaque image est mesurée sur une grille réduite (media/shot_cutter.h), la vidéo découpée
// en plans, et les plans finis écrits dans une ShotTimeline que le grade adaptatif lit.
//
// Décodage logiciel seul, comme les images fixes du SPAN FP8 (stills.h) : la lecture garde le GPU et son
// décodeur. L'analyse va bien plus vite que la lecture (une H.264 1080p : environ 25 fois sur 4 cœurs) et se
// met en attente quand elle a assez d'avance.
#pragma once

#include <atomic>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

#include "core/frame.h"
#include "core/types.h"
#include "media/shot_cutter.h"
#include "media/shot_timeline.h"

namespace ajp::media {

// Les mesures d'une image décodée en mémoire hôte (Nv12, Yuv420p, P010, Yuv420p10 ; BT.601 ou BT.709) :
// une grille de 128 × 72 points, hors bandes noires. Refus FormatUnsupported pour une image HDR, d'un autre
// format ou hors de la mémoire hôte.
std::expected<ShotFrame, Refusal> measure_frame(const DecodedFrame& frame);

struct ShotProbeOptions {
  int threads = 0;  // threads du décodeur ; 0 : la moitié des cœurs, 2 à 4 (1 sous Windows)
  std::int64_t lead_us = 180'000'000;      // avance au-delà de laquelle l'analyse attend
  std::int64_t skip_known_us = 5'000'000;  // zone déjà analysée plus longue : sautée
};

class ShotProbe {
 public:
  // Analyse file depuis start_us dans timeline, qui doit survivre à la ShotProbe.
  static std::unique_ptr<ShotProbe> start(std::filesystem::path file, ShotTimeline& timeline,
                                          std::int64_t start_us, ShotProbeOptions options = {});

  ShotProbe(const ShotProbe&) = delete;
  ShotProbe& operator=(const ShotProbe&) = delete;
  ~ShotProbe();  // demande l'arrêt et attend le thread

  // L'instant affiché : l'analyse se cale sur lui pour son avance.
  void set_playhead(std::int64_t us);
  // Un saut de la lecture : l'analyse repart de at_us, sauf si cet endroit est déjà analysé.
  void seek(std::int64_t at_us);
  // L'analyse a atteint la fin du fichier, ou s'est arrêtée sur un refus (failure()).
  bool finished() const;
  std::optional<Refusal> failure() const;
  // La dernière image mesurée.
  std::int64_t frontier_us() const;

 private:
  ShotProbe(std::filesystem::path file, ShotTimeline& timeline, ShotProbeOptions options);
  void run(const std::stop_token& stop, std::int64_t start_us);
  void fail(Refusal refusal);

  std::filesystem::path file_;
  ShotTimeline& timeline_;
  ShotProbeOptions options_;
  std::atomic<std::int64_t> playhead_{0};
  std::atomic<std::int64_t> seek_to_{-1};
  std::atomic<std::int64_t> frontier_{0};
  std::atomic<bool> finished_{false};
  mutable std::mutex failure_mutex_;
  std::optional<Refusal> failure_;
  std::jthread thread_;
};

}  // namespace ajp::media
