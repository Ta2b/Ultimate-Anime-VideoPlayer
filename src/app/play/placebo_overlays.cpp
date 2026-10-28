// Calques de PlaceboScreen : l'atlas R8 des sous-titres et celui de l'OSD, téléversés
// seulement quand ils changent, dessinés en monochrome coloré (subs/subtitle_frame.h).
#include <algorithm>
#include <format>
#include <utility>

#include "app/play/controls.h"
#include "app/play/placebo_impl.h"

namespace ajp::app::play {

Overlays::~Overlays() {
  pl_tex_destroy(gpu_, &subtitles_.tex);
  pl_tex_destroy(gpu_, &osd_.tex);
}

std::expected<bool, Refusal> Overlays::add(Layer& layer, const subs::SubtitleFrame& frame, OutputSize out,
                                           int ui_bottom) {
  if (!frame.atlas || frame.rects.empty() || frame.width <= 0 || frame.height <= 0) return false;
  const subs::SubtitleAtlas& atlas = *frame.atlas;
  if (layer.atlas != &atlas || layer.generation != atlas.generation) {
    pl_tex_params params{};
    params.w = atlas.width;
    params.h = atlas.height;
    params.format = pl_find_named_fmt(gpu_, "r8");
    params.sampleable = true;
    params.host_writable = true;
    if (!pl_tex_recreate(gpu_, &layer.tex, &params)) {
      return std::unexpected(
          Refusal{RefusalCode::VendorCallFailed,
                  std::format("atlas {}x{} : pl_tex_recreate", atlas.width, atlas.height)});
    }
    pl_tex_transfer_params upload{};
    upload.tex = layer.tex;
    upload.row_pitch = static_cast<std::size_t>(atlas.width);
    upload.ptr = const_cast<std::uint8_t*>(atlas.pixels.data());  // NOLINT : lu seulement
    if (!pl_tex_upload(gpu_, &upload)) {
      return std::unexpected(Refusal{RefusalCode::VendorCallFailed, "atlas : pl_tex_upload"});
    }
    layer.atlas = &atlas;
    layer.generation = atlas.generation;
  }
  // Les rectangles sont placés pour la taille demandée ; une fenêtre redimensionnée depuis les
  // étire à la taille courante.
  const float sx = static_cast<float>(out.width) / static_cast<float>(frame.width);
  const float sy = static_cast<float>(out.height) / static_cast<float>(frame.height);
  layer.parts.clear();
  for (const subs::SubtitleRect& rect : frame.rects) {
    pl_overlay_part part{};
    part.src = {static_cast<float>(rect.src.x), static_cast<float>(rect.src.y),
                static_cast<float>(rect.src.x + rect.src.width),
                static_cast<float>(rect.src.y + rect.src.height)};
    part.dst = {static_cast<float>(rect.dst_x) * sx, static_cast<float>(rect.dst_y) * sy,
                static_cast<float>(rect.dst_x + rect.src.width) * sx,
                static_cast<float>(rect.dst_y + rect.src.height) * sy};
    part.color[0] = static_cast<float>(rect.color.r) / 255.0F;
    part.color[1] = static_cast<float>(rect.color.g) / 255.0F;
    part.color[2] = static_cast<float>(rect.color.b) / 255.0F;
    part.color[3] = static_cast<float>(rect.color.a) / 255.0F;
    layer.parts.push_back(part);
  }
  // Les sous-titres de la moitié basse montent au-dessus de la barre de lecture visible.
  if (ui_bottom > 0) {
    const float height = static_cast<float>(out.height);
    float bottom = 0.0F;
    for (const pl_overlay_part& part : layer.parts) {
      if (part.dst.y0 >= height * 0.5F) bottom = std::max(bottom, part.dst.y1);
    }
    const float lift = subtitle_lift(bottom, height, static_cast<float>(ui_bottom));
    for (pl_overlay_part& part : layer.parts) {
      if (lift > 0.0F && part.dst.y0 >= height * 0.5F) {
        part.dst.y0 -= lift;
        part.dst.y1 -= lift;
      }
    }
  }
  pl_overlay& overlay = overlays_[static_cast<std::size_t>(count_++)];
  overlay = {};
  overlay.tex = layer.tex;
  overlay.mode = PL_OVERLAY_MONOCHROME;
  overlay.coords = PL_OVERLAY_COORDS_DST_FRAME;
  overlay.repr = pl_color_repr_rgb;
  overlay.color = pl_color_space_srgb;
  overlay.parts = layer.parts.data();
  overlay.num_parts = static_cast<int>(layer.parts.size());
  return true;
}

std::expected<std::span<const pl_overlay>, Refusal> Overlays::build(const subs::SubtitleFrame* subtitles,
                                                                    const std::span<const std::string>* osd,
                                                                    OsdText& text, OutputSize out,
                                                                    int ui_bottom, int ui_top) {
  count_ = 0;
  if (subtitles != nullptr) {
    if (auto added = add(subtitles_, *subtitles, out, ui_bottom); !added)
      return std::unexpected(std::move(added.error()));
  }
  if (osd != nullptr) {
    auto frame = text.render(*osd, out);
    if (!frame) return std::unexpected(std::move(frame.error()));
    if (*frame != nullptr) {
      if (auto added = add(osd_, **frame, out); !added) return std::unexpected(std::move(added.error()));
      for (pl_overlay_part& part : osd_.parts) {  // sous le titre de l'interface
        part.dst.y0 += static_cast<float>(ui_top);
        part.dst.y1 += static_cast<float>(ui_top);
      }
    }
  }
  return std::span<const pl_overlay>(overlays_.data(), static_cast<std::size_t>(count_));
}

}  // namespace ajp::app::play
