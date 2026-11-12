// L'écran de la fenêtre suivi pendant la lecture : sa fréquence, son pic et son HDR, son VRR armé, et le
// passage d'un écran à l'autre. Sorti de run_chain (B08-14). Les relevés coûteux se font sur le fil de
// DisplayProbe (B11-05, B11-06, B32-02, B08-09) ; ici, sur le thread main, ce qui en découle.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>

#include <SDL3/SDL_video.h>

#include "app/play/auto_config.h"
#include "app/play/chain.h"
#include "app/play/display_contract.h"
#include "app/play/display_probe.h"
#include "app/play/player_engines.h"
#include "app/play/session.h"
#include "app/play/vrr_watch.h"

namespace spdlog {
class logger;
}

namespace ajp::app::play {

// Ce que le mode Auto lit de l'écran (run_chain : apply_auto).
struct DisplayState {
  // La fréquence de l'écran, gardée quand le système ne la donne plus : un écran rallumé passe un moment sans
  // mode courant (0 Hz), et le plafond de l'auto retombait au rythme de la source.
  double known_hz = 0.0;
  std::int64_t known_period_us = 0;
  bool vrr_armed = false;
};

class DisplayFollow {
 public:
  struct Hooks {
    std::function<void()> apply_auto;   // le plafond et le mode de l'Auto suivent l'écran
    std::function<void()> apply_floor;  // le plancher du VRR suit son armement
  };
  DisplayFollow(Player& p, Chain& chain, Session& session, DisplayState& state, double source_fps,
                Hooks hooks, std::shared_ptr<spdlog::logger> log, void (*wake)(void*), void* context);

  // Toutes les kDisplayPollUs (ou quand le VRR change) : le contrat d'écran, un relevé demandé au fil, et le
  // changement d'écran.
  void poll();
  // À chaque tour : le relevé fini, s'il y en a un (fréquence, pic et HDR, VRR armé).
  void take();

 private:
  void follow_peak(std::optional<double> peak_now);
  void follow_vrr(std::optional<bool> reading);
  void follow_window(const DisplaySdl& sdl, bool display_changed);
  void follow_display_facts(const std::optional<DisplayFacts>& display);

  Player& p_;
  Chain& chain_;
  Session& session_;
  DisplayState& state_;
  Hooks hooks_;
  std::shared_ptr<spdlog::logger> log_;
  SDL_DisplayID known_display_ = 0;
  double session_hz_seen_ = 0.0;
  DisplayMove opened_;
  ContractWatch contract_watch_;
  VrrWatch vrr_watch_;
  bool vrr_fullscreen_ = true;  // relu au premier relevé : VRR, compositeur (P3.45)
  std::int64_t hdr_polled_at_ = 0;
  bool display_moved_ = false;  // la fenêtre a changé d'écran : le relevé de l'écran demandé le suit
  DisplayProbe probe_;          // en dernier : son fil s'arrête avant le reste
};

}  // namespace ajp::app::play
