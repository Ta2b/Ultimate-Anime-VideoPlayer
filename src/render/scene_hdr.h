// Les métadonnées HDR10+ d'un plan (8.9) : la source HDR10 annonce, plan par plan, sa lumière mesurée par la
// pré-analyse (ShotMeta), et libplacebo la ramène à la cible du contrat d'écran par sa spline, sans
// détection de pic : un plan sombre reste tel quel, un plan qui dépasse la cible est compressé.
#pragma once

#include <libplacebo/colorspace.h>
#include <libplacebo/shaders/colorspace.h>

namespace ajp::render {

struct SceneHdr {
  float max_nits = 0.0F;  // le maximum des composantes (maxSCL), en nits
  float avg_nits = 0.0F;  // la moyenne du plan, en nits
  friend bool operator==(const SceneHdr&, const SceneHdr&) = default;
};

// Vrai si ces valeurs se lisent comme des métadonnées HDR10+ (maximum et moyenne positifs).
bool usable(const SceneHdr& scene);

// scene_max (les trois composantes) et scene_avg de l'image ; rien si les valeurs ne se lisent pas.
void set_scene_hdr(pl_color_space& image, const SceneHdr& scene);

// La conversion qui lit ces métadonnées : spline de libplacebo, métadonnées HDR10+ seules.
pl_color_map_params scene_color_map(const pl_color_map_params& base = pl_color_map_default_params);

}  // namespace ajp::render
