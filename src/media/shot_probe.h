// La pré-analyse des plans : un second décodage du fichier lu, vidéo seule, en tâche de fond et en avance
// sur la lecture. Chaque image est mesurée sur une grille réduite (media/shot_cutter.h), la vidéo découpée
// en plans, et les plans finis écrits dans une ShotTimeline que le grade lit.
//
// Décodage logiciel seul, comme les images fixes du SPAN FP8 (stills.h) : la lecture garde le GPU et son
// décodeur. L'analyse va bien plus vite que la lecture (une H.264 1080p : environ 25 fois sur 4 cœurs) et se
// met en attente quand elle a assez d'avance.
#pragma once

#include <chrono>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include "core/frame.h"
#include "core/types.h"
#include "media/shot_cutter.h"
#include "media/shot_timeline.h"

namespace ajp::media {

// Les mesures d'une image décodée en mémoire hôte (Nv12, Yuv420p, P010, Yuv420p10 ; BT.601, BT.709 ou
// BT.2020 ; SDR, PQ ou HLG) : la grille de kernels/spec/frame_stats.h, hors bandes noires, mesurée par
// kernels/ref/frame_stats.h. Refus FormatUnsupported pour un autre format ou hors de la mémoire hôte.
std::expected<ShotFrame, Refusal> measure_frame(const DecodedFrame& frame);

struct ShotProbeOptions {
  int threads = 0;  // threads du décodeur ; 0 : la moitié des cœurs, 2 à 4 (1 sous Windows)
  std::int64_t lead_us = 180'000'000;      // avance au-delà de laquelle l'analyse attend
  std::int64_t skip_known_us = 5'000'000;  // zone déjà analysée plus longue : sautée
  // --probe-nvdec (8.8) : les images décodées par NVDEC à cette taille (DecoderOptions::cuvid_resize) ;
  // vide : le décodeur logiciel en pleine taille.
  std::string cuvid_resize;
  // Ce que le grade fait à la lumière de la source, pour les séquences de flashs que l'analyse marque dans la
  // chronologie (media/flash_guard.h).
  double flash_gain = 1.0;
  // Sur le thread de l'analyse, avant elle : la part déjà connue de la vidéo (le cache). Vrai si la vidéo
  // est connue entière : rien n'est analysé. La lecture du cache quitte ainsi le thread de la fenêtre
  // (A42-14 : un disque en veille y retardait l'ouverture).
  std::function<bool(ShotTimeline&)> preload;
  // Sur le thread de l'analyse, quand elle s'arrête (arrêt demandé, refus) : la part faite, à garder au
  // cache. complete : la vidéo est analysée entière, sans refus. L'écriture quitte ainsi le thread de la
  // fenêtre (B42-05).
  std::function<void(const ShotTimeline&, bool complete)> on_stop;
  // B30-01 : le thread d'une analyse qui ne s'arrête pas dans stop_limit à la destruction (une lecture
  // suspendue à un partage réseau muet, que le drapeau d'interruption de FFmpeg n'atteint pas) est confié à
  // abandon ; il ne touche plus qu'à l'état qu'il partage (la chronologie comprise). stopped dit s'il a
  // fini. Vide : la destruction attend le thread.
  std::function<void(std::jthread thread, std::function<bool()> stopped)> abandon;
  std::chrono::milliseconds stop_limit{2000};
};
inline constexpr const char* kProbeCuvidSize = "256x144";

class ShotProbe {
 public:
  // Analyse file depuis start_us dans timeline, que l'analyse partage.
  static std::unique_ptr<ShotProbe> start(std::filesystem::path file, std::shared_ptr<ShotTimeline> timeline,
                                          std::int64_t start_us, ShotProbeOptions options = {});

  ShotProbe(const ShotProbe&) = delete;
  ShotProbe& operator=(const ShotProbe&) = delete;
  ShotProbe(ShotProbe&&) = delete;
  ShotProbe& operator=(ShotProbe&&) = delete;
  // Demande l'arrêt et attend le thread : au plus stop_limit si abandon est donné (ShotProbeOptions).
  ~ShotProbe();

  // L'instant affiché : l'analyse se cale sur lui pour son avance.
  void set_playhead(std::int64_t us);
  // Un saut de la lecture : l'analyse repart de at_us, sauf si cet endroit est déjà analysé.
  void seek(std::int64_t at_us);
  // L'analyse a atteint la fin du fichier, ou s'est arrêtée sur un refus (failure()).
  bool finished() const;
  std::optional<Refusal> failure() const;
  // La dernière image mesurée.
  std::int64_t frontier_us() const;

  struct State;  // l'état que le thread partage (shot_probe.cpp)

 private:
  explicit ShotProbe(std::shared_ptr<State> state);
  std::shared_ptr<State> state_;
  std::jthread thread_;
};

}  // namespace ajp::media
