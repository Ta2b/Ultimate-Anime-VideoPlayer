// La taille de l'interface : l'échelle du bureau, ou celle choisie dans Général > Application.
// GNOME laisse un écran 2560×1440 de 24 pouces à 100 % là où Windows propose 125 % : le menu y paraissait
// minuscule et crénelé. Hors Windows et macOS, dont l'échelle est
// celle que le système recommande, l'automatique agrandit donc un grand écran laissé à 100 %.
#pragma once

#include <algorithm>

#include "core/profile.h"

namespace kirari::app::play {

// La taille choisie, en pour cent ; 0 : automatique.
inline constexpr core::Range<int> kUiScalePercent{100, 200};

// L'échelle automatique, d'après celle du bureau et la hauteur de l'écran en pixels. trust_system : le
// système choisit lui-même une échelle recommandée (Windows, macOS), qu'on garde telle quelle.
constexpr float auto_ui_scale(float system, int display_height, bool trust_system) {
  if (!(system > 0.0F)) system = 1.0F;
  if (trust_system || system > 1.0F) return system;
  if (display_height >= 2160) return 1.5F;
  if (display_height >= 1440) return 1.25F;
  return system;
}

// L'échelle de l'interface : percent (kUiScalePercent) si l'utilisateur en a choisi une, l'automatique
// sinon.
constexpr float ui_scale(int percent, float system, int display_height, bool trust_system) {
  if (percent > 0)
    return static_cast<float>(std::clamp(percent, kUiScalePercent.lo, kUiScalePercent.hi)) / 100.0F;
  return auto_ui_scale(system, display_height, trust_system);
}

}  // namespace kirari::app::play
