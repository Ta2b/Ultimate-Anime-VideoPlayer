// Calques de PlaceboScreen : l'atlas R8 des sous-titres et celui de l'OSD, téléversés
// seulement quand ils changent, dessinés en monochrome coloré (subs/subtitle_frame.h).
#include <algorithm>
#include <cmath>
#include <format>
#include <utility>
#include <vector>

#include "app/play/controls.h"
#include "app/play/placebo_impl.h"

namespace ajp::app::play {

Overlays::~Overlays() {
  pl_tex_destroy(gpu_, &subtitles_.tex);
  pl_tex_destroy(gpu_, &osd_.tex);
  pl_tex_destroy(gpu_, &tint_.tex);
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
  // Les dialogues du bas montent au-dessus de la barre de lecture visible : libass les place ainsi dès que
  // la demande porte la montée (frame.line_lift, A24-19). Le temps que cette image arrive, celle rendue
  // sans montée est décalée ici, panneaux de la moitié basse compris.
  if (ui_bottom > 0 && frame.line_lift == 0.0F) {
    const float height = static_cast<float>(out.height);
    std::vector<std::pair<float, float>> spans;
    spans.reserve(layer.parts.size());
    for (const pl_overlay_part& part : layer.parts) spans.emplace_back(part.dst.y0, part.dst.y1);
    // Seul le paquet collé au bas de l'image monte (C24-03) : un panneau lié au décor reste en place.
    const std::vector<bool> lifted = bottom_cluster(spans, height);
    float bottom = 0.0F;
    for (std::size_t i = 0; i < spans.size(); ++i) {
      if (lifted[i]) bottom = std::max(bottom, spans[i].second);
    }
    const float lift = subtitle_lift(bottom, height, static_cast<float>(ui_bottom));
    for (std::size_t i = 0; i < layer.parts.size(); ++i) {
      if (lift > 0.0F && lifted[i]) {
        layer.parts[i].dst.y0 -= lift;
        layer.parts[i].dst.y1 -= lift;
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

// Le débogage des paires : un voile sur toute la cible et un cadre de quatre bandes, d'une texture 1 × 1
// pleine que la couleur de chaque partie teinte (PL_OVERLAY_MONOCHROME).
std::expected<void, Refusal> Overlays::add_tint(const PairTint& tint, OutputSize out) {
  if (out.width <= 0 || out.height <= 0) return {};
  if (tint_.tex == nullptr) {
    pl_tex_params params{};
    params.w = 1;
    params.h = 1;
    params.format = pl_find_named_fmt(gpu_, "r8");
    params.sampleable = true;
    params.host_writable = true;
    const std::uint8_t full = 255;
    params.initial_data = &full;
    if (!pl_tex_recreate(gpu_, &tint_.tex, &params))
      return std::unexpected(Refusal{RefusalCode::VendorCallFailed, "teinte des paires : pl_tex_recreate"});
  }
  const auto w = static_cast<float>(out.width);
  const auto h = static_cast<float>(out.height);
  const float band = std::max(4.0F, std::round(std::min(w, h) * 0.012F));
  const auto part = [&](pl_rect2df dst, const std::array<float, 4>& color) {
    pl_overlay_part p{};
    p.src = {0.0F, 0.0F, 1.0F, 1.0F};
    p.dst = dst;
    std::copy(color.begin(), color.end(), p.color);
    tint_.parts.push_back(p);
  };
  tint_.parts.clear();
  part({0.0F, 0.0F, w, h}, tint.wash);
  part({0.0F, 0.0F, w, band}, tint.frame);
  part({0.0F, h - band, w, h}, tint.frame);
  part({0.0F, band, band, h - band}, tint.frame);
  part({w - band, band, w, h - band}, tint.frame);
  pl_overlay& overlay = overlays_[static_cast<std::size_t>(count_++)];
  overlay = {};
  overlay.tex = tint_.tex;
  overlay.mode = PL_OVERLAY_MONOCHROME;
  overlay.coords = PL_OVERLAY_COORDS_DST_FRAME;
  overlay.repr = pl_color_repr_rgb;
  overlay.color = pl_color_space_srgb;
  overlay.parts = tint_.parts.data();
  overlay.num_parts = static_cast<int>(tint_.parts.size());
  return {};
}

std::expected<Overlays::Built, Refusal> Overlays::build(const subs::SubtitleFrame* subtitles,
                                                        const std::span<const std::string>* osd,
                                                        OsdText& text, OutputSize out, int ui_bottom,
                                                        int ui_top, const std::optional<PairTint>& tint) {
  count_ = 0;
  if (subtitles != nullptr) {
    if (auto added = add(subtitles_, *subtitles, out, ui_bottom); !added)
      return std::unexpected(std::move(added.error()));
  }
  const auto subtitle_count = static_cast<std::size_t>(count_);
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
  if (tint) {  // sous l'OSD : le texte reste lisible
    if (auto added = add_tint(*tint, out); !added) return std::unexpected(std::move(added.error()));
    if (count_ > static_cast<int>(subtitle_count) + 1)
      std::rotate(overlays_.begin() + static_cast<std::ptrdiff_t>(subtitle_count),
                  overlays_.begin() + count_ - 1, overlays_.begin() + count_);
  }
  const std::span<const pl_overlay> all(overlays_.data(), static_cast<std::size_t>(count_));
  return Built{.all = all, .subtitles = all.first(subtitle_count), .osd = all.subspan(subtitle_count)};
}

}  // namespace ajp::app::play
