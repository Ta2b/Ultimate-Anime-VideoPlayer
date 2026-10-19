// Couche des sous-titres : le SubtitleFrame de subs (atlas R8 de couverture, rectangles,
// couleurs) devient une pl_overlay monochrome que pl_render_image dessine sur la cible, après
// le grade. Interne à render.
//
// L'atlas est téléversé dans une pl_tex R8 réutilisée, seulement quand son pointeur ou sa
// génération change ; la couche garde le shared_ptr du dernier atlas (contrat de
// subs/subtitle_frame.h), qui rentre chez subs quand le suivant le remplace.
#pragma once

#include <expected>
#include <memory>
#include <span>
#include <vector>

#include <libplacebo/renderer.h>

#include "core/types.h"
#include "subs/subtitle_frame.h"

namespace ajp::render {

class SubtitleLayer {
 public:
  explicit SubtitleLayer(pl_gpu gpu);
  SubtitleLayer(const SubtitleLayer&) = delete;
  SubtitleLayer& operator=(const SubtitleLayer&) = delete;
  SubtitleLayer(SubtitleLayer&&) = delete;
  SubtitleLayer& operator=(SubtitleLayer&&) = delete;
  ~SubtitleLayer();

  // L'overlay de frame, placé sur le rectangle de l'image dans la cible (PL_OVERLAY_COORDS_DST_CROP),
  // à l'échelle crop / (frame.width, frame.height). hdr_white_nits > 0 : blanc des sous-titres
  // dans une cible HDR (subtitle_white) ; 0 : cible SDR. Vide quand il n'y a rien à dessiner.
  // Valide jusqu'au prochain appel. InputMismatch : rectangle hors de l'atlas ou taille nulle ;
  // RenderFailed : texture ou téléversement refusés.
  std::expected<std::span<const pl_overlay>, Refusal> prepare(const subs::SubtitleFrame& frame,
                                                              const pl_rect2df& crop, float hdr_white_nits);

  // Rend l'atlas tenu à subs (fin de lecture, seek).
  void drop_atlas();

 private:
  std::expected<void, Refusal> upload(const subs::SubtitleAtlas& atlas);

  pl_gpu gpu_;
  pl_tex tex_ = nullptr;
  std::shared_ptr<const subs::SubtitleAtlas> atlas_;
  std::uint64_t atlas_generation_ = 0;
  std::vector<pl_overlay_part> parts_;
  pl_overlay overlay_{};
};

}  // namespace ajp::render
