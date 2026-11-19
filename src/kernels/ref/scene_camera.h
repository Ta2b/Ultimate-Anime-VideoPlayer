// Référence de l'affinage de la caméra de la carte (kernels/spec/scene_camera.h), et ce que le lecteur en
// tire. scene_camera_refine fixe le contrat du port CUDA (gpu/cuda/backend/scene_camera_kernels.cu) et sert
// aux tests et au banc (carte_rejeu) ; scene_camera_request et scene_camera_apply sont du code du lecteur
// (app/play/scene_map.cpp), comme scene_motion_of.
#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include "kernels/image.h"
#include "kernels/ref/scene_flow.h"
#include "kernels/spec/scene_camera.h"

namespace kirari::kernels::ref {

// L'affinage d'une paire, sur ses lumas du flux (W × H, de même taille) : SizeMismatch si les tuiles du décor
// ne sont pas scene_flow_tiles(W) × scene_flow_tiles(H).
Result<SceneCameraRefined> scene_camera_refine(const Plane<std::uint8_t>& a, const Plane<std::uint8_t>& b,
                                               const SceneCameraRequest& request);

// La demande d'affinage d'une caméra : rien si elle n'est pas ajustée, ou si son décor a moins de
// scene::kRefineMinPixels pixels.
std::optional<SceneCameraRequest> scene_camera_request(const SceneCamera& camera);

// La caméra affinée : son modèle (si au moins un pas a été fait), la vérification de chaque tuile (checks),
// et l'écart mesuré sur l'image, médiane des décalages qui restent sur les tuiles du décor résolues dans les
// deux sens (scene::kCheckMinTiles au moins), sinon sur celles résolues dans un sens (un bord : seul l'écart
// en travers se mesure, et c'est le seul qui se voit). Une tuile compte si au moins scene::kCheckCover de ses
// pixels l'ont été.
void scene_camera_apply(SceneCamera& camera, const SceneCameraRefined& refined);

// La caméra d'une paire en pixels de travail, telle que scene_motion_of la pose (SceneMotion::camera,
// camera_refined, residual, residual_tiles) : identité si elle n'est pas ajustée. L'image du flux fait
// flow_width × flow_height ; divisor va du travail au flux (SceneFlowTiles).
struct SceneCameraWork {
  std::array<double, 6> camera{0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  bool fit = false;
  bool refined = false;
  double residual = 0.0;
  int residual_tiles = 0;
};
SceneCameraWork scene_camera_work(const SceneCamera& camera, int flow_width, int flow_height, int divisor);

// Le plus grand écart entre deux modèles (coordonnées réduites, pixels du flux) aux quatre coins de l'image
// du flux, pixels du flux.
double scene_camera_gap(const std::array<double, 6>& a, const std::array<double, 6>& b, int flow_width,
                        int flow_height);

}  // namespace kirari::kernels::ref
