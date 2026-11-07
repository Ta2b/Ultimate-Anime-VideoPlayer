// La configuration automatique (docs/ux-parametres.md, 4.7) : un préréglage (Auto, Économe, Rapide,
// Normal, Qualité, Ultra, Personnalisé) donne la chaîne (réduction, SPAN, interpolation) selon la
// tranche de la carte (gpu_tier.h), l'écran et la vidéo. « Personnalisé » ne touche à rien : c'est
// l'état des réglages faits à la main, jamais écrasé. Seule la chaîne est écrite ; le HDR, les
// couleurs, le son et les langues restent à l'utilisateur.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "app/play/gpu_tier.h"
#include "app/play/settings.h"
#include "core/display.h"

namespace ajp::app::play {

// DistilDRBA dans l'Auto (tranches Rapide et Ultra) : faux tant que sa porte de qualité n'est pas passée
// (docs/adr/0003) ; ces tranches prennent alors RIFE 4.22 lite, comme les deux du dessous. Le choix manuel
// de DistilDRBA reste ouvert.
inline constexpr bool kDistilDrbaInAuto = false;

// ArtCNN C4F32 (kLightSpanModel, P6.7) dans les tranches Économe et Léger : faux tant que le banc sur iGPU
// n'a pas montré une meilleure image qu'Économe (sans agrandissement) au même coût (docs/phase6-a-tester.md).
// Vrai, Économe agrandit avec lui au lieu de jouer la source telle quelle, et Léger le prend à la place de
// SPAN. Le choix manuel dans la liste des modèles reste ouvert.
inline constexpr bool kLightSpanInAuto = false;

std::string_view key(
    Preset preset);  // « auto », « economy », « fast », « normal », « quality », « ultra », « custom »
std::string_view label(Preset preset);  // « Auto », « Économe », « Rapide », « Normal », « Qualité », « Ultra
                                        // », « Personnalisé »
std::optional<Preset> preset_from_key(std::string_view text);
std::string_view key(Priority priority);    // « fluidity », « balance », « image »
std::string_view label(Priority priority);  // « Fluidité », « Équilibre », « Image »
std::optional<Priority> priority_from_key(std::string_view text);

// La machine : sa tranche détectée et l'écran. display_height et display_hz à 0 : inconnus (1080p, 60 Hz).
// Ce que la calibration de l'assistant a mesuré sur cette carte et cet écran (calibration.toml) : en Auto,
// la chaîne garde ces bornes et s'adapte encore à chaque vidéo (source, cadence, écran).
struct CalibratedChain {
  bool span = true;
  bool rife = true;
  int height = 0;          // la plus haute résolution de travail que SPAN tient
  std::string span_model;  // vide : le modèle par défaut (FP16) ; sinon celui joué en FP8
};

struct Machine {
  Tier tier = Tier::Normal;
  int display_height = 0;
  double display_hz = 0.0;
  // Ce que l'exploration du benchmark a trouvé de mieux que la chaîne du niveau (bench.toml) ; 0 ou vide : la
  // chaîne du niveau telle quelle. Ne valent que pour le préréglage Auto.
  int explored_work_height = 0;
  std::string explored_rife_model;
  std::optional<CalibratedChain> calibrated;  // Auto calibré : prime sur l'exploration et le niveau
};

// Le niveau de chaîne d'un préréglage : Auto prend la tranche de la machine ; Économe, Rapide, Normal,
// Qualité et Ultra donnent les niveaux 0 à 4. Personnalisé : rien.
std::optional<Tier> chain_level(Preset preset, Tier detected);

// settings avec la chaîne du préréglage pour cette machine et cette source ; settings.preset ==
// Custom : inchangé. Idempotente.
PlaySettings apply_preset(const PlaySettings& settings, const Machine& machine,
                          const core::SourceInfo& source);

// La chaîne (celle que apply_preset écrit) diffère-t-elle entre a et b ? Un réglage de chaîne fait à la main
// pendant qu'un préréglage est choisi bascule les réglages en Personnalisé.
bool preset_chain_differs(const PlaySettings& a, const PlaySettings& b);

// P3.21 : la fenêtre passée sur un écran d'une autre hauteur rouvre la chaîne d'un préréglage (la hauteur de
// travail suit l'écran courant) ; Personnalisé garde la sienne.
bool display_moves_chain(Preset preset, int opened_height, int now_height);

}  // namespace ajp::app::play
