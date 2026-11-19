// Référence de la carte de la scène (kernels/spec/scene_flow.h). scene_luma et scene_tile_sums fixent le
// contrat du port CUDA (gpu/cuda/backend/scene_flow_kernels.cu) et ne servent qu'aux tests ; scene_motion_of
// est du code du lecteur, comme frame_stats_of : il tire la carte des sommes du noyau, avec l'historique des
// classes que le lecteur garde d'une paire à la suivante.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "kernels/image.h"
#include "kernels/spec/scene_camera.h"
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
// de classe que vue deux paires de suite (sinon tous les clients clignotent) ; entre les deux, la classe
// tenue reste, sous la confiance sûre des clients (scene::kContradicted). Une paire où la tuile ne change pas
// sous une caméra tenue ne compte pas : un dessin tenu (image doublée d'une animation en deux ou en trois)
// garde sa classe, un effet au plus scene::kEffectIdlePairs paires (idle les compte). Vide, ou d'une autre
// grille : la paire part de ses propres classes.
struct SceneHistory {
  int cols = 0;
  int rows = 0;
  std::vector<SceneClass> held;
  std::vector<SceneClass> seen;
  std::vector<std::uint8_t> idle;  // paires de suite où la tuile n'a pas compté
  void clear() {
    cols = 0;
    rows = 0;
    held.clear();
    seen.clear();
    idle.clear();
  }
};

// La caméra d'une paire, ajustée sur les vecteurs moyens des tuiles (la méthode du banc 2.1 : départ sur la
// translation médiane, kFitRounds tours de moindres carrés pondérés par le coût, rejet au-delà de 3 écarts
// médians bornés entre kFitBoundLow et kFitBoundHigh pixels à 720p), puis, si le backend sait, affinée par
// Lucas-Kanade sur les lumas (spec/scene_camera.h, ref::scene_camera_apply).
struct SceneCamera {
  bool fit = false;
  std::array<double, 6> model{};  // déplacement en coordonnées réduites (spec/scene_camera.h), pixels du flux
  double residual = 0.0;          // écart moyen pondéré des vecteurs des tuiles gardées au modèle, du flux
  std::vector<std::uint8_t> decor;  // cols × rows, 1 : tuile gardée par l'ajustement
  int decor_pixels = 0;             // leurs pixels du flux
  bool refined = false;             // affinée par Lucas-Kanade (au moins un pas)
  double check = -1.0;              // écart mesuré sur l'image, pixels du flux ; négatif : pas mesuré
  int check_tiles = 0;              // tuiles de cette mesure
  // Par tuile, cols × rows : l'écart après la caméra mesuré sur l'image (spec/scene_camera.h), dont
  // scene_motion_of sépare le dessin du décor. Vide sans affinage relu.
  std::vector<SceneTileCheck> checks;
};

// La caméra des vecteurs de NVOF, avant l'affinage ; pas ajustée sur une coupe probable.
SceneCamera scene_camera_of(const SceneFlowTiles& tiles);

// La carte d'une paire, avec sa caméra (scene_camera_of, affinée ou non). history est mis à jour ; il est
// vidé après une coupe probable.
SceneMotion scene_motion_of(const SceneFlowTiles& tiles, SceneHistory& history, const SceneCamera& camera);
// Sans affinage : scene_motion_of(tiles, history, scene_camera_of(tiles)).
SceneMotion scene_motion_of(const SceneFlowTiles& tiles, SceneHistory& history);

}  // namespace kirari::kernels::ref
