// Référence de la carte de la scène (kernels/spec/scene_flow.h). scene_luma et scene_tile_sums fixent le
// contrat du port CUDA (gpu/cuda/backend/scene_flow_kernels.cu) et ne servent qu'aux tests ; scene_motion_of
// est du code du lecteur, comme frame_stats_of : il tire la carte des sommes du noyau, avec l'historique des
// classes que le lecteur garde d'une paire à la suivante.
#pragma once

#include <cstdint>
#include <vector>

#include "kernels/image.h"
#include "kernels/spec/scene_flow.h"

namespace kirari::kernels::ref {

// La luma 8 bits du flux : W / d × H / d codes. DimensionOutOfRange si l'image du flux est sous
// kSceneFlowMinSide.
Result<Plane<std::uint8_t>> scene_luma(const RgbImage& rgb, int divisor);

// Un vecteur de NVOF, en S10.5 (1/32 de pixel du flux), et son coût.
struct FlowVector {
  std::int16_t x = 0;
  std::int16_t y = 0;
};
// Les sommes par tuile. flow et cost : scene_flow_cells(largeur) × scene_flow_cells(hauteur) ; a et b de même
// taille. DimensionOutOfRange sinon.
Result<SceneFlowTiles> scene_tile_sums(const Plane<std::uint8_t>& a, const Plane<std::uint8_t>& b,
                                       const Plane<FlowVector>& flow, const Plane<std::uint8_t>& cost,
                                       int divisor);

// Les classes des paires précédentes, par tuile : la classe tenue et la dernière vue. Une tuile ne change
// de classe que vue deux paires de suite (sinon tous les clients clignotent). Vide, ou d'une autre grille :
// la paire part de ses propres classes.
struct SceneHistory {
  int cols = 0;
  int rows = 0;
  std::vector<SceneClass> held;
  std::vector<SceneClass> seen;
  void clear() {
    cols = 0;
    rows = 0;
    held.clear();
    seen.clear();
  }
};

// La carte d'une paire. history est mis à jour ; il est vidé après une coupe probable.
SceneMotion scene_motion_of(const SceneFlowTiles& tiles, SceneHistory& history);

}  // namespace kirari::kernels::ref
