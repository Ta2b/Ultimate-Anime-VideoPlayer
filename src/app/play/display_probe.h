// Les relevés de l'écran faits hors du thread qui présente (B11-05, B11-06, B32-02, B08-09). Sous Windows, la
// fréquence passe par QueryDisplayConfig, le pic par une fabrique DXGI et l'énumération des sorties, le VRR
// armé par NVAPI : des millisecondes, parfois des dizaines sur un portable hybride, qui faisaient manquer des
// présentations à 144 Hz deux fois par seconde et toutes les 10 s ; l'EDID et NVAPI de l'écran
// (window_display) à chaque bascule du plein écran (C11-06). Le thread main lit ce que seuls les appels vidéo
// de SDL disent (display_sdl, quelques µs), un fil les complète, et la boucle prend le résultat au tour où il
// arrive (réveillée par `wake`).
//
// Un seul relevé à la fois : une demande faite pendant qu'un relevé tourne remplace la précédente en attente.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>

#include <SDL3/SDL_video.h>

#include "app/play/host_facts.h"

namespace ajp::app::play {

struct DisplayReading {
  double hz = 0.0;                      // la fréquence de l'écran ; 0 : inconnue
  bool peak_read = false;               // le pic a été demandé, et relu
  std::optional<ContentPeak> peak;      // la crête vue par le contenu et sa source (content_peak)
  bool vrr_read = false;                // le VRR armé a été demandé, et relu
  std::optional<bool> vrr_armed;        // window_vrr_armed
  bool display_read = false;            // l'écran a été demandé, et relu (C11-06)
  std::optional<DisplayFacts> display;  // window_display
  std::int64_t took_us = 0;             // le temps du relevé, pour le journal
};

// Ce que le fil relit : les fonctions de host_facts.h par défaut, des doublures dans les tests.
struct DisplayReaders {
  std::function<double(SDL_Window*, const DisplaySdl&)> hz;
  std::function<std::optional<ContentPeak>(SDL_Window*, const DisplaySdl&)> peak;
  std::function<std::optional<bool>(SDL_Window*, const std::string&)> vrr;
  std::function<std::optional<DisplayFacts>(SDL_Window*, const DisplaySdl&)> display;
};
DisplayReaders system_display_readers();

class DisplayProbe {
 public:
  // window vit plus longtemps que la sonde ; wake(context) est appelé du fil quand un relevé est prêt.
  DisplayProbe(SDL_Window* window, DisplayReaders readers, void (*wake)(void*) = nullptr,
               void* context = nullptr);
  ~DisplayProbe();
  DisplayProbe(const DisplayProbe&) = delete;
  DisplayProbe& operator=(const DisplayProbe&) = delete;

  // Un relevé : la fréquence toujours, le pic si `peak`, le VRR armé si `connector` n'est pas vide (Linux) ou
  // si `vrr` (Windows, NVAPI), l'écran de la fenêtre (EDID, NVAPI) si `display`. Thread main.
  void ask(DisplaySdl sdl, bool peak, bool vrr, std::string connector, bool display = false);
  // Le dernier relevé fini, une fois ; rien sinon. Thread main.
  std::optional<DisplayReading> take();
  // Un relevé est demandé ou tourne.
  bool busy() const;

 private:
  struct Request {
    DisplaySdl sdl;
    bool peak = false;
    bool vrr = false;
    std::string connector;
    bool display = false;
  };
  void run(const std::stop_token& stop);

  SDL_Window* window_;
  DisplayReaders readers_;
  void (*wake_)(void*);
  void* context_;
  mutable std::mutex mutex_;
  std::condition_variable_any changed_;
  std::optional<Request> pending_;
  bool running_ = false;
  std::optional<DisplayReading> done_;
  std::jthread thread_;  // en dernier : arrêté avant le reste
};

}  // namespace ajp::app::play
