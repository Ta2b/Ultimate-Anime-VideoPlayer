// Le flux optique matériel de NVIDIA (NVOF) dans la pré-analyse des plans (media/shot_probe.h) : pour chaque
// image, son mouvement face à l'image d'avant, sur la luma réduite à kProbeFlowWidth × kProbeFlowHeight.
// Le découpeur (media/shot_cutter.h) en tire un signal de coupe : sur une vraie coupe, presque aucun bloc ne
// retrouve son voisin ; sur un panoramique, un éclair ou un geste vif, la plupart le retrouvent.
//
// Deux parties : motion_of, pur calcul sur le processeur (les vecteurs et les deux lumas donnent la part de
// blocs sans correspondance), testé sans GPU ; ProbeFlow, la session NVOF (bibliothèque du pilote chargée à
// l'exécution, comme media/cuda_driver.h), qui rend rien sans carte NVIDIA, sans pilote ou sur un refus :
// le découpeur se comporte alors exactement comme avant.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "core/frame.h"
#include "media/probe_motion.h"

namespace kirari::media {

// La luma de la pré-analyse : le format de NVDEC réduit (kProbeCuvidSize, 256 × 144). Une image d'une autre
// taille (décodage logiciel en pleine taille) y est ramenée par moyenne de zones.
inline constexpr int kProbeFlowWidth = 256;
inline constexpr int kProbeFlowHeight = 144;
inline constexpr int kProbeFlowGrid = 4;  // un vecteur par bloc de 4 × 4 (la seule grille des Turing)
inline constexpr int kProbeFlowCellsX = kProbeFlowWidth / kProbeFlowGrid;
inline constexpr int kProbeFlowCellsY = kProbeFlowHeight / kProbeFlowGrid;

// Une luma 8 bits de kProbeFlowWidth × kProbeFlowHeight, lignes contiguës.
using ProbeLuma = std::vector<std::uint8_t>;

// Un vecteur de NVOF, en S10.5 (1/32 de pixel), comme le matériel les rend (NV_OF_FLOW_VECTOR).
struct FlowVector {
  std::int16_t x = 0;
  std::int16_t y = 0;
};

// Le mouvement de current face à previous (deux ProbeLuma), d'après les vecteurs de NVOF (kProbeFlowCellsX ×
// kProbeFlowCellsY, l'entrée current, la référence previous). Un bloc est apparié si son contenu, centré
// (sans sa moyenne : un éclair ou un fondu qui décale la lumière ne compte pas), se retrouve dans previous au
// bout de son vecteur avec un écart sous la moitié de son propre détail. Le sens du vecteur est essayé dans
// les deux signes et nul : la convention de NVOF n'a pas à être devinée juste. Rien si moins d'un bloc sur
// vingt est texturé (une image unie, un blanc d'éclair) : pas de quoi juger.
std::optional<ProbeMotion> motion_of(std::span<const std::uint8_t> previous,
                                     std::span<const std::uint8_t> current,
                                     std::span<const FlowVector> vectors);

// La luma de frame (mémoire hôte ; Nv12, Yuv420p, P010, P016, Yuv420p10, Yuv444p16) ramenée en ProbeLuma ;
// rien pour un autre format.
std::optional<ProbeLuma> probe_luma_of(const DecodedFrame& frame);

// La session NVOF de la pré-analyse, sur le fil de l'analyse seulement (l'API n'est pas sûre entre fils).
// Sans indices temporels : le résultat d'une paire ne dépend pas de l'historique (sauts, cache, export).
class ProbeFlow {
 public:
  // Ouvre la session sur l'appareil CUDA d'ordinal device ; rien (et une ligne de journal) sans carte NVIDIA,
  // sans flux optique dans le pilote, ou sur un refus.
  static std::unique_ptr<ProbeFlow> open(int device = 0);
  ~ProbeFlow();
  ProbeFlow(const ProbeFlow&) = delete;
  ProbeFlow& operator=(const ProbeFlow&) = delete;
  ProbeFlow(ProbeFlow&&) = delete;
  ProbeFlow& operator=(ProbeFlow&&) = delete;

  // Le mouvement de frame face à l'image poussée juste avant ; rien pour la première après forget(), pour un
  // format non lu, ou sur un refus de NVOF (la session se ferme alors pour de bon).
  std::optional<ProbeMotion> push(const DecodedFrame& frame);
  // Une rupture de la suite d'images (un saut de l'analyse) : la prochaine image n'a pas de précédente.
  void forget();

  struct Session;  // la session NVOF et ses tampons (probe_flow_nvof.cpp)

 private:
  explicit ProbeFlow(std::unique_ptr<Session> session);
  std::unique_ptr<Session> session_;
  std::optional<ProbeLuma> previous_;
};

}  // namespace kirari::media
