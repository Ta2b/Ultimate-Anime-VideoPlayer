// Ce que l'ouverture de la chaîne tire des réglages et qui ne se refait pas en direct : la profondeur de
// l'anneau et l'ordre du warp et de SPAN (qui décide aussi la variante mouvement de DistilDRBA). Un réglage
// qui les change rouvre la chaîne (change_between, settings.h), sinon l'interpolation tournerait dans un
// anneau trop court ou avec un SPAN par image créée (A08-03).
#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "app/play/i18n.h"
#include "app/play/options.h"
#include "app/play/settings.h"
#include "core/types.h"

namespace ajp::app::play {

// Le plus haut facteur d'interpolation que la lecture peut prendre : celui choisi à la main, ou en Auto le
// palier entier le plus haut que l'écran affiche (core/auto_ceiling), jusqu'à rife_factor ; 1 sans
// interpolation ou écran inconnu en Auto. C'est le plafond que le menu affiche (« Auto va jusqu'à ×N »).
int top_factor(const PlaySettings& settings, double display_hz, double source_fps);
// Le facteur qui décide l'ordre du warp et de SPAN : celui choisi à la main, 0 (inconnu) en Auto. Le facteur
// d'Auto varie avec la charge et avec l'écran (×5 à 120 Hz, ×10 à 240 Hz) : en faire la règle de « après SPAN
// » donnait la voie rapide, aux défauts de warp agrandis, sur presque tout écran rapide, sans mesure de
// qualité (B22-03). En Auto, seule la hauteur de sortie décide (kWarpAfterSpanAbove, mesurée en temps).
int order_factor(const PlaySettings& settings, double display_hz, double source_fps);
// Emplacements du ring : 4 images, 6 de ×3 à ×4, 8 au-delà, et toujours 8 en Rapide (le facteur suit
// l'écran).
std::uint32_t ring_depth(const PlaySettings& settings, Factor factor);

struct ChainShape {
  std::uint32_t ring = 4;
  bool warp_after_span = false;
  bool operator==(const ChainShape&) const = default;
};

// Une chaîne ouverte : sa forme, et ce dont elle a été tirée (l'écran et la source de l'ouverture, pour
// qu'un réglage se compare à la même mesure).
struct OpenedChain {
  // Ce que le moteur d'interpolation a été chargé pour : les réglages de l'ouverture. Coupée, l'interpolation
  // laisse changer ces réglages en direct (rien ne se recharge, live_part), mais le moteur chargé reste
  // celui-ci : change_between compare le choix à lui quand l'interpolation revient (B22-01).
  struct Loaded {
    MotionEngine motion_engine = MotionEngine::Rife;
    std::string rife_model;
    PrecisionTier motion_precision = PrecisionTier::Balanced;
    SpanOrder span_order = SpanOrder::Auto;
    bool rife_fused = true;
    bool drba_fused = true;
    bool operator==(const Loaded&) const = default;
  };

  ChainShape shape;
  double display_hz = 0.0;  // 0 : inconnue
  double source_fps = 0.0;
  int output_height = 0;     // lignes après SPAN
  bool span = false;         // SPAN chargé : sans lui, l'ordre du warp ne change rien
  bool rife_loaded = false;  // sans moteur d'interpolation, l'anneau reste à 4
  // DistilDRBA tel que la chaîne l'a pris (A23-07) : le panneau le dit, au lieu de laisser croire que la case
  // « fusionné » décide seule.
  bool drba = false;                // DistilDRBA est le moteur du mouvement chargé
  bool drba_fused = false;          // ses quatre moteurs fusionnés tournent
  bool drba_motion = false;         // la variante mouvement (un SPAN par image source)
  bool drba_fused_capable = false;  // le backend sait le fusionner
  // Un moteur fusionné demandé, essayé, puis refusé (A03-21) : le réseau entier tourne à sa place, plus
  // lent. Le code du refus, pour le dire à l'écran et au panneau ; le détail est au journal.
  std::optional<RefusalCode> drba_fused_refused;
  std::optional<RefusalCode> rife_fused_refused;
  std::optional<Loaded> loaded;  // rien : on compare au réglage en cours
};

// Les réglages de l'ouverture que le moteur d'interpolation a chargés (OpenedChain::loaded).
OpenedChain::Loaded loaded_by(const PlaySettings& settings);

// L'avis de l'ouverture quand un moteur fusionné a été refusé ; vide sinon.
i18n::Translated fused_refusal_notice(const OpenedChain& opened);

// La forme que ces réglages demandent pour l'écran et la source de at.
ChainShape chain_shape(const PlaySettings& settings, const OpenedChain& at);
// Ces réglages tournent-ils dans la chaîne ouverte ? Sans interpolation, toujours. Un anneau plus profond que
// demandé convient ; l'ordre du warp doit être le même, dans les deux sens (avant SPAN est le choix de
// qualité sous ×3, le garder après serait une baisse de qualité sans mesure).
bool fits(const OpenedChain& opened, const PlaySettings& wanted);

}  // namespace ajp::app::play
