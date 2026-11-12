// La pré-analyse des plans : un second décodage du fichier lu, vidéo seule, en tâche de fond et en avance
// sur la lecture. Chaque image est mesurée sur une grille réduite (media/shot_cutter.h), la vidéo découpée
// en plans, et les plans finis écrits dans une ShotTimeline que le grade lit.
//
// Décodage par NVDEC en image réduite (256 × 144, --probe-nvdec, le défaut ; un décodeur à part de celui de
// la lecture), ou logiciel en pleine taille quand NVDEC ne prend pas le format (H.264 10 bits) ou manque.
// L'analyse va à pleine vitesse jusqu'à quelques secondes d'avance, puis à quelques fois le temps réel
// (C42-01), et se met en attente quand elle a assez d'avance.
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
  // Threads du décodage logiciel ; 0 : la moitié des cœurs, 2 à 4 (background_decoder_threads). NVDEC réduit
  // n'en prend qu'un.
  int threads = 0;
  std::int64_t lead_us = 180'000'000;  // avance au-delà de laquelle l'analyse attend
  // Le rythme au-delà de pace_after_us d'avance : pace fois le temps réel (0 : pleine vitesse jusqu'à
  // lead_us).
  double pace = 4.0;
  std::int64_t pace_after_us = 20'000'000;
  std::int64_t skip_known_us = 5'000'000;  // zone déjà analysée plus longue : sautée
  // --probe-nvdec (8.8) : les images décodées par NVDEC à cette taille (DecoderOptions::cuvid_resize) ;
  // vide : le décodeur logiciel en pleine taille.
  std::string cuvid_resize;
  // La peau des visages (media/face_skin.h, option face_skin) : chaque plan garde la palette de ses visages.
  bool faces = false;
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
  // fini. C42-06 : la destruction n'attend que la fin de l'analyse, pas on_stop ; un thread encore dans
  // on_stop (l'écriture du cache, avec son fsync) est confié à abandon de même, sans attendre. Vide : la
  // destruction attend le thread.
  std::function<void(std::jthread thread, std::function<bool()> stopped)> abandon;
  std::chrono::milliseconds stop_limit{2000};
};
inline constexpr const char* kProbeCuvidSize = "256x144";
inline constexpr const char* kProbeCuvidFacesSize = "512x288";  // la recherche des visages (face_skin)

class ShotProbe {
 public:
  // Analyse file depuis start_us dans timeline, que l'analyse partage.
  static std::unique_ptr<ShotProbe> start(std::filesystem::path file, std::shared_ptr<ShotTimeline> timeline,
                                          std::int64_t start_us, ShotProbeOptions options = {});

  ShotProbe(const ShotProbe&) = delete;
  ShotProbe& operator=(const ShotProbe&) = delete;
  ShotProbe(ShotProbe&&) = delete;
  ShotProbe& operator=(ShotProbe&&) = delete;
  // Demande l'arrêt et attend la fin de l'analyse : au plus stop_limit si abandon est donné
  // (ShotProbeOptions), sans attendre on_stop.
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
