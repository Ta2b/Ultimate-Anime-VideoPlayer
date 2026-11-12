// La carte qui pilote l'écran de la fenêtre, comparée à celle du rendu (C37-07). Sur un portable hybride
// (Optimus sans MUX, PRIME), l'écran est branché sur l'iGPU : Vulkan présente depuis la RTX et le système
// recopie chaque image vers l'iGPU. Pas de G-SYNC, un retard et de la bande passante PCIe en plus ; le mode
// « dGPU seul » du BIOS ou de l'outil du constructeur l'évite.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "app/play/i18n.h"
#include "core/gpu_device.h"

struct SDL_Window;

namespace ajp::app::play {

struct ScreenCard {
  std::string name;      // « Intel(R) UHD Graphics » sous Windows ; « PCI 8086:a7a0 » sous Linux
  bool renders = false;  // la carte du rendu pilote aussi l'écran
};

// Linux : la carte d'un connecteur DRM (« card1-DP-2 » : card1) ; vide s'il n'en nomme pas. Pur.
std::string drm_card_of(std::string_view connector);

// La carte de l'écran de la fenêtre ; rien si elle ne se lit pas (macOS, une seule carte, sysfs absent).
// connector : celui de DisplayFacts (Linux). Thread de la fenêtre.
std::optional<ScreenCard> screen_card(SDL_Window* window, std::string_view connector,
                                      const GpuDevice& render);

// La phrase du journal : « écran sur la carte X, rendu sur Y ». Pure.
std::string screen_card_line(const ScreenCard& card, std::string_view render_name);

// L'écart vaut un avertissement à l'écran quand la recopie coûte : 4K et plus, ou 144 Hz et plus. Pur.
constexpr bool screen_card_warns(const ScreenCard& card, int display_height, double display_hz) {
  return !card.renders && (display_height >= 2160 || display_hz >= 143.5);
}

// À l'ouverture de la chaîne : la carte de l'écran au journal, et une phrase pour l'écran la première fois
// que la recopie coûte (C37-07) ; Smooth Motion demandé sur une carte ou un pilote qui ne le fait pas, au
// journal (C37-03).
std::optional<i18n::Translated> note_render_device(SDL_Window* window, std::string_view connector,
                                                   const GpuDevice& render, bool smooth_motion);

}  // namespace ajp::app::play
