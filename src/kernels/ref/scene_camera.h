// Référence de l'affinage de la caméra de la carte (kernels/spec/scene_camera.h), et ce que le lecteur en
// tire. scene_camera_refine fixe le contrat du port CUDA (gpu/cuda/backend/scene_camera_kernels.cu) et sert
// aux tests et au banc (carte_rejeu) ; scene_camera_request et scene_camera_apply sont du code du lecteur
// (app/play/scene_map.cpp), comme scene_motion_of.
#pragma once

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

// La caméra affinée : son modèle (si au moins un pas a été fait), et l'écart mesuré sur l'image, médiane des
// décalages qui restent sur les tuiles résolues dans les deux sens (scene::kCheckMinTiles au moins), sinon
// sur celles résolues dans un sens (un bord : seul l'écart en travers se mesure, et c'est le seul qui se
// voit). Une tuile compte si au moins scene::kCheckCover de ses pixels l'ont été.
void scene_camera_apply(SceneCamera& camera, const SceneCameraRefined& refined);

}  // namespace kirari::kernels::ref
