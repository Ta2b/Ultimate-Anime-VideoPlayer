// La configuration automatique (docs/ux-parametres.md, 4.7) : un préréglage (Auto, Économe, Rapide,
// Normal, Qualité, Ultra, Personnalisé) donne la chaîne (réduction, SPAN, interpolation) selon la
// tranche de la carte (gpu_tier.h), l'écran et la vidéo. « Personnalisé » ne touche à rien : c'est
// l'état des réglages faits à la main, jamais écrasé. Seule la chaîne est écrite ; le HDR, les
// couleurs, le son et les langues restent à l'utilisateur.
#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

#include "app/play/gpu_tier.h"
#include "app/play/settings.h"
#include "core/resolve.h"

namespace ajp::app::play {

std::string_view key(Preset preset);  // « auto », « economy », « fast », « normal », « quality », « ultra », « custom »
std::string_view label(Preset preset);  // « Auto », « Économe », « Rapide », « Normal », « Qualité », « Ultra », « Personnalisé »
std::optional<Preset> preset_from_key(std::string_view text);
std::string_view key(Priority priority);  // « fluidity », « balance », « image »
std::string_view label(Priority priority);  // « Fluidité », « Équilibre », « Image »
std::optional<Priority> priority_from_key(std::string_view text);

// La machine : sa tranche détectée et l'écran. display_height et display_hz à 0 : inconnus (1080p, 60 Hz).
struct Machine {
  Tier tier = Tier::Normal;
  int display_height = 0;
  double display_hz = 0.0;
};

// Le niveau de chaîne d'un préréglage : Auto prend la tranche de la machine ; Économe, Rapide, Normal,
// Qualité et Ultra donnent les niveaux 0 à 4. Personnalisé : rien.
std::optional<Tier> chain_level(Preset preset, Tier detected);

// settings avec la chaîne du préréglage pour cette machine et cette source ; settings.preset ==
// Custom : inchangé. Idempotente.
PlaySettings apply_preset(const PlaySettings& settings, const Machine& machine, const core::SourceInfo& source);

// La chaîne (celle que apply_preset écrit) diffère-t-elle entre a et b ? Un réglage de chaîne fait à la main
// pendant qu'un préréglage est choisi bascule les réglages en Personnalisé.
bool preset_chain_differs(const PlaySettings& a, const PlaySettings& b);

}  // namespace ajp::app::play
