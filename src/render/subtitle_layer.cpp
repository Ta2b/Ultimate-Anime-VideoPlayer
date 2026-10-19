#include "render/subtitle_layer.h"

#include <algorithm>
#include <cstddef>
#include <format>

#include <libplacebo/gpu.h>

#include "render/c_structs.h"

namespace ajp::render {
namespace {

bool inside(const subs::AtlasBox& box, const subs::SubtitleAtlas& atlas) {
  return box.x >= 0 && box.y >= 0 && box.width > 0 && box.height > 0 && box.x <= atlas.width - box.width &&
         box.y <= atlas.height - box.height;
}

float unit(std::uint8_t value) {
  return static_cast<float>(value) / 255.0F;
}

}  // namespace

SubtitleLayer::SubtitleLayer(pl_gpu gpu) : gpu_(gpu) {}

SubtitleLayer::~SubtitleLayer() {
  pl_tex_destroy(gpu_, &tex_);
}

void SubtitleLayer::drop_atlas() {
  atlas_.reset();
  atlas_generation_ = 0;
}

std::expected<void, Refusal> SubtitleLayer::upload(const subs::SubtitleAtlas& atlas) {
  const auto size = static_cast<std::size_t>(atlas.width) * static_cast<std::size_t>(atlas.height);
  if (atlas.width <= 0 || atlas.height <= 0 || atlas.pixels.size() != size) {
    return std::unexpected(
        Refusal{RefusalCode::InputMismatch,
                std::format("atlas {}x{} de {} octets", atlas.width, atlas.height, atlas.pixels.size())});
  }
  // La texture grandit au besoin et ne rapetisse jamais : un atlas plus petit en réutilise un coin.
  const int width = tex_ != nullptr ? std::max(tex_->params.w, atlas.width) : atlas.width;
  const int height = tex_ != nullptr ? std::max(tex_->params.h, atlas.height) : atlas.height;
  auto params = detail::zeroed<pl_tex_params>();
  params.w = width;
  params.h = height;
  params.format = pl_find_named_fmt(gpu_, "r8");
  params.sampleable = true;
  params.host_writable = true;
  params.debug_tag = "ajp sous-titres";
  if (params.format == nullptr || !pl_tex_recreate(gpu_, &tex_, &params)) {
    return std::unexpected(Refusal{RefusalCode::RenderFailed, "pl_tex_recreate (atlas des sous-titres)"});
  }
  pl_tex_transfer_params transfer{};
  transfer.tex = tex_;
  transfer.rc = {.x0 = 0, .y0 = 0, .z0 = 0, .x1 = atlas.width, .y1 = atlas.height, .z1 = 1};
  transfer.row_pitch = static_cast<std::size_t>(atlas.width);
  transfer.ptr = const_cast<std::uint8_t*>(atlas.pixels.data());
  if (!pl_tex_upload(gpu_, &transfer)) {
    return std::unexpected(Refusal{RefusalCode::RenderFailed, "pl_tex_upload (atlas des sous-titres)"});
  }
  return {};
}

std::expected<std::span<const pl_overlay>, Refusal> SubtitleLayer::prepare(const subs::SubtitleFrame& frame,
                                                                           const pl_rect2df& crop,
                                                                           float hdr_white_nits) {
  if (frame.atlas == nullptr) {
    drop_atlas();  // rien à ce pts : l'atlas tenu rentre chez subs
    return std::span<const pl_overlay>{};
  }
  if (frame.rects.empty()) return std::span<const pl_overlay>{};
  if (frame.width <= 0 || frame.height <= 0) {
    return std::unexpected(Refusal{RefusalCode::InputMismatch,
                                   std::format("sous-titres placés sur {}x{}", frame.width, frame.height)});
  }
  if (frame.atlas != atlas_ || frame.atlas->generation != atlas_generation_) {
    if (auto ok = upload(*frame.atlas); !ok) return std::unexpected(ok.error());
    atlas_ = frame.atlas;
    atlas_generation_ = frame.atlas->generation;
  }

  const float sx = (crop.x1 - crop.x0) / static_cast<float>(frame.width);
  const float sy = (crop.y1 - crop.y0) / static_cast<float>(frame.height);
  parts_.clear();
  for (const subs::SubtitleRect& rect : frame.rects) {
    if (!inside(rect.src, *atlas_)) {
      return std::unexpected(
          Refusal{RefusalCode::InputMismatch,
                  std::format("rectangle {}x{} en ({}, {}) hors de l'atlas {}x{}", rect.src.width,
                              rect.src.height, rect.src.x, rect.src.y, atlas_->width, atlas_->height)});
    }
    const auto x = static_cast<float>(rect.dst_x);
    const auto y = static_cast<float>(rect.dst_y);
    const auto w = static_cast<float>(rect.src.width);
    const auto h = static_cast<float>(rect.src.height);
    parts_.push_back(pl_overlay_part{
        .src = {.x0 = static_cast<float>(rect.src.x),
                .y0 = static_cast<float>(rect.src.y),
                .x1 = static_cast<float>(rect.src.x) + w,
                .y1 = static_cast<float>(rect.src.y) + h},
        .dst = {.x0 = x * sx, .y0 = y * sy, .x1 = (x + w) * sx, .y1 = (y + h) * sy},
        .color = {unit(rect.color.r), unit(rect.color.g), unit(rect.color.b), unit(rect.color.a)},
    });
  }

  // Couverture R8 × couleur de chaque morceau, alpha droit, en sRGB (contrat de subs) ; dans
  // une cible HDR, le blanc sRGB vaut subtitle_white, comme sub-hdr-peak de mpv.
  pl_color_space color = pl_color_space_srgb;
  if (hdr_white_nits > 0.0F) color.hdr.max_luma = hdr_white_nits;
  pl_color_repr repr = pl_color_repr_rgb;
  repr.alpha = PL_ALPHA_INDEPENDENT;
  overlay_ = pl_overlay{.tex = tex_,
                        .mode = PL_OVERLAY_MONOCHROME,
                        .coords = PL_OVERLAY_COORDS_DST_CROP,
                        .repr = repr,
                        .color = color,
                        .parts = parts_.data(),
                        .num_parts = static_cast<int>(parts_.size())};
  return std::span<const pl_overlay>(&overlay_, 1);
}

}  // namespace ajp::render
