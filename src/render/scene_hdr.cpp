#include "render/scene_hdr.h"

#include <libplacebo/tone_mapping.h>

namespace ajp::render {

bool usable(const SceneHdr& scene) {
  return scene.max_nits > 0.0F && scene.avg_nits > 0.0F;
}

void set_scene_hdr(pl_color_space& image, const SceneHdr& scene) {
  if (!usable(scene)) return;
  for (float& c : image.hdr.scene_max) c = scene.max_nits;
  // Une moyenne au-dessus du maximum ne se lit pas : libplacebo la prendrait pour la lumière du plan.
  image.hdr.scene_avg = scene.avg_nits < scene.max_nits ? scene.avg_nits : scene.max_nits;
}

pl_color_map_params scene_color_map(const pl_color_map_params& base) {
  pl_color_map_params params = base;
  params.tone_mapping_function = &pl_tone_map_spline;
  params.metadata = PL_HDR_METADATA_HDR10PLUS;
  return params;
}

}  // namespace ajp::render
