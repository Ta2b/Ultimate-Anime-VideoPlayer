// Le contrat d'écran lu et relu (chantier 8.2, render::DisplayContract) : le blanc de référence et la crête
// de KWin, la marge HDR de SDL, l'EDID (primaires, noir, OLED) et, quand une séance de sonde l'a mesurée, la
// courbe du limiteur. Relu quand SDL signale un changement d'état HDR (SDL_EVENT_WINDOW_HDR_STATE_CHANGED)
// ou quand KDE réécrit kwinoutputconfig.json : KDE qui assombrit l'écran fait passer la référence à 83.
// Le contrat se lit et se journalise ; le grade et l'indication à la swapchain ne s'en servent qu'au contrat
// v2 (8.4, éteint).
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <spdlog/logger.h>

#include "app/play/display_edid.h"
#include "app/play/system_peak.h"
#include "render/display_model.h"

struct SDL_Window;

namespace ajp::app::play {

// La courbe du limiteur relevée à la sonde (8.12), cd/m² aux surfaces render::kAblAreas.
struct MeasuredAbl {
  std::array<double, 5> nits{};
};

render::DisplayContract contract_of(const DisplayFacts& facts, const std::optional<KwinOutput>& kwin,
                                    double headroom, const std::optional<MeasuredAbl>& abl);

// « référence 265 nits (k 1,305), pic 1000 nits (766 du contenu), MaxFALL 261 (200 du contenu), noir
// 0,0006, OLED, primaires R 0,684/0,305 G 0,250/0,715 B 0,147/0,062 »
std::string describe(const render::DisplayContract& contract);

// La relecture : tick() à chaque tour de la boucle d'affichage (ou toutes les 500 ms), qui rend le contrat
// quand il a changé (et le journalise).
class ContractWatch {
 public:
  ContractWatch(SDL_Window* window, std::optional<DisplayFacts> facts, std::shared_ptr<spdlog::logger> log);
  ~ContractWatch();
  ContractWatch(const ContractWatch&) = delete;
  ContractWatch& operator=(const ContractWatch&) = delete;

  // Ce que le filtre d'événements de SDL pose (display_contract.cpp).
  struct Flags {
    std::atomic<bool>* hdr;
    std::atomic<bool>* peak;
  };
  std::optional<render::DisplayContract> tick();
  // La fenêtre est passée sur un autre écran (A11-01) : son connecteur et son EDID remplacent ceux de
  // l'ouverture, et le contrat est relu au prochain tick. Sans cela, KWin restait lu pour l'ancien
  // connecteur, avec les primaires, le noir, l'OLED et le pic de repli de l'ancien écran.
  void set_display(std::optional<DisplayFacts> facts) {
    facts_ = std::move(facts);
    hdr_changed_.store(true, std::memory_order_relaxed);
    peak_changed_.store(true, std::memory_order_relaxed);
  }
  // Vrai une fois après un événement qui peut changer le HDR ou le pic du système : HDR allumé ou coupé,
  // fenêtre passée sur un autre écran, réglages de KDE réécrits. Le relevé du pic (énumérations d'écrans
  // coûteuses sous Windows, A11-02) se fait alors tout de suite, au lieu d'un sondage toutes les 2 s.
  bool take_peak_change() { return peak_changed_.exchange(false, std::memory_order_relaxed); }
  const std::optional<render::DisplayContract>& current() const { return contract_; }

 private:
  render::DisplayContract read() const;

  SDL_Window* window_;
  std::optional<DisplayFacts> facts_;
  std::shared_ptr<spdlog::logger> log_;
  std::optional<render::DisplayContract> contract_;
  std::optional<std::int64_t> kwin_stamp_;
  std::atomic<bool> hdr_changed_{true};    // posé par le filtre d'événements SDL ; vrai : première lecture
  std::atomic<bool> peak_changed_{false};  // les mêmes événements, pour take_peak_change
  Flags flags_{&hdr_changed_, &peak_changed_};
};

}  // namespace ajp::app::play
