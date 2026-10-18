#include "subs/subtitle_renderer.h"

#include <algorithm>
#include <cstddef>
#include <format>
#include <span>

#include <ass/ass.h>

#include "subs/atlas_packer.h"
#include "subs/timebase.h"

namespace ajp::subs {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

bool side_in_range(int side) {
  return 1 <= side && side <= kMaxTargetSide;
}

std::expected<void, Refusal> validate(const RenderRequest& request) {
  const RenderTarget& t = request.target;
  const bool video_known = t.video_width != 0 || t.video_height != 0;
  if (!side_in_range(t.width) || !side_in_range(t.height) ||
      (video_known && (!side_in_range(t.video_width) || !side_in_range(t.video_height)))) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("sous-titres : sortie {}×{}, vidéo {}×{}", t.width, t.height, t.video_width,
                              t.video_height));
  }
  if (!timestamp_in_range(request.pts_us)) {
    return refuse(RefusalCode::FieldOutOfRange, std::format("sous-titres : pts {} µs", request.pts_us));
  }
  return {};
}

// Couleur libass 0xRRGGBBTT, TT la transparence, en opacité droite.
Rgba8 color_of(std::uint32_t rgbt) {
  return {.r = static_cast<std::uint8_t>(rgbt >> 24),
          .g = static_cast<std::uint8_t>(rgbt >> 16),
          .b = static_cast<std::uint8_t>(rgbt >> 8),
          .a = static_cast<std::uint8_t>(255U - (rgbt & 0xFFU))};
}

std::size_t to_size(int value) {
  return static_cast<std::size_t>(value);
}

// Copie la couverture de chaque image à sa place empaquetée.
void copy_pieces(std::span<const ass_image* const> pieces, const Packing& packing, SubtitleAtlas& atlas) {
  atlas.width = packing.width;
  atlas.height = packing.height;
  atlas.pixels.assign(to_size(packing.width) * to_size(packing.height), 0);
  for (std::size_t i = 0; i < pieces.size(); ++i) {
    const ass_image& image = *pieces[i];
    const std::size_t width = to_size(image.w);
    const std::size_t stride = to_size(image.stride);
    // La dernière ligne n'est garantie que sur w octets.
    const std::span<const unsigned char> source(image.bitmap, stride * (to_size(image.h) - 1) + width);
    for (std::size_t row = 0; row < to_size(image.h); ++row) {
      const std::size_t target =
          (to_size(packing.positions[i].y) + row) * to_size(packing.width) + to_size(packing.positions[i].x);
      std::ranges::copy(source.subspan(row * stride, width),
                        atlas.pixels.begin() + static_cast<std::ptrdiff_t>(target));
    }
  }
}

}  // namespace

void SubtitleRenderer::Free::operator()(ass_renderer* renderer) const {
  ass_renderer_done(renderer);
}

SubtitleRenderer::SubtitleRenderer(Handle renderer, ass_library* library, RendererOptions options)
    : renderer_(std::move(renderer)), library_(library), options_(std::move(options)) {}

std::expected<SubtitleRenderer::Handle, Refusal> SubtitleRenderer::make(ass_library* library,
                                                                        const RendererOptions& options) {
  if (options.render_threads < 0 || options.render_threads > kMaxRenderThreads) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("sous-titres : {} threads de rendu", options.render_threads));
  }
  Handle renderer(ass_renderer_init(library));
  if (renderer == nullptr) return refuse(RefusalCode::LibassFailed, "ass_renderer_init");

  const std::string default_font = options.default_font.string();
  const int provider =
      options.fonts == FontProvider::System ? ASS_FONTPROVIDER_AUTODETECT : ASS_FONTPROVIDER_NONE;
  ass_set_fonts(renderer.get(), default_font.empty() ? nullptr : default_font.c_str(),
                options.default_family.c_str(), provider, nullptr, 1);
  ass_set_render_thread_count(renderer.get(), options.render_threads);
  return renderer;
}

std::expected<SubtitleRenderer, Refusal> SubtitleRenderer::create(const AssLibrary& library,
                                                                  const RendererOptions& options) {
  auto renderer = make(library.get(), options);
  if (!renderer) return std::unexpected(std::move(renderer.error()));
  return SubtitleRenderer(std::move(*renderer), library.get(), options);
}

void SubtitleRenderer::fonts_changed() {
  stale_fonts_ = rendered_;
}

std::expected<void, Refusal> SubtitleRenderer::refresh_fonts() {
  auto fresh = make(library_, options_);
  if (!fresh) return std::unexpected(std::move(fresh.error()));
  forget();
  renderer_ = std::move(*fresh);
  target_.reset();
  rendered_ = false;
  stale_fonts_ = false;
  return {};
}

void SubtitleRenderer::configure(const RenderTarget& target) {
  ass_set_frame_size(renderer_.get(), target.width, target.height);
  const bool video_known = target.video_width != 0;
  ass_set_storage_size(renderer_.get(), video_known ? target.video_width : target.width,
                       video_known ? target.video_height : target.height);
  target_ = target;
  in_sync_ = false;
}

void SubtitleRenderer::forget() {
  current_.reset();
  rects_.clear();
  in_sync_ = false;
  refused_ = false;
}

std::unexpected<Refusal> SubtitleRenderer::refuse_content(Refusal refusal) {
  in_sync_ = true;  // libass comparera l'image suivante à ce contenu refusé
  refused_ = true;
  return std::unexpected(std::move(refusal));
}

std::expected<RenderOutcome, Refusal> SubtitleRenderer::render(const SubtitleTrack& track,
                                                               const RenderRequest& request,
                                                               AtlasPool& pool) {
  if (auto valid = validate(request); !valid) return std::unexpected(std::move(valid.error()));
  if (stale_fonts_) {
    if (auto refreshed = refresh_fonts(); !refreshed) return std::unexpected(std::move(refreshed.error()));
  }
  if (target_ != request.target) configure(request.target);
  if (const bool vsfilter = vsfilter_converts(track.script_matrix(), request.target.video_matrix);
      vsfilter != vsfilter_) {
    vsfilter_ = vsfilter;
    in_sync_ = false;  // mêmes couvertures, autres couleurs
  }
  rendered_ = true;

  int change = 2;
  pieces_.clear();
  for (const ass_image* image =
           ass_render_frame(renderer_.get(), track.get(), to_ms(request.pts_us), &change);
       image != nullptr; image = image->next) {
    if (image->w > 0 && image->h > 0) pieces_.push_back(image);  // libass : taille nulle, rien à dessiner
  }

  SubtitleFrame frame{.pts_us = request.pts_us,
                      .epoch = request.epoch,
                      .width = request.target.width,
                      .height = request.target.height};
  if (pieces_.empty()) {
    forget();
    in_sync_ = true;  // libass compare l'image suivante à une liste vide, et nous aussi
    return frame;
  }
  // change 0 : mêmes images ; 1 : mêmes couvertures, déplacées ; 2 : contenu neuf.
  const bool unchanged = in_sync_ && change < 2;
  if (unchanged && refused_) return frame;  // refusé une fois, à son apparition
  if (!unchanged || current_ == nullptr || pieces_.size() != rects_.size()) {
    return build(std::move(frame), pool);
  }
  if (change == 1) {
    for (std::size_t i = 0; i < pieces_.size(); ++i) {
      rects_[i].dst_x = pieces_[i]->dst_x;
      rects_[i].dst_y = pieces_[i]->dst_y;
    }
  }
  frame.atlas = current_;
  frame.rects = rects_;
  return frame;
}

std::expected<RenderOutcome, Refusal> SubtitleRenderer::build(SubtitleFrame frame, AtlasPool& pool) {
  forget();  // relâcher l'atlas courant d'abord : il est peut-être le dernier libre
  if (pieces_.size() > kMaxRectsPerFrame) {
    return refuse_content({RefusalCode::SubtitleAtlasFull,
                           std::format("sous-titres à {} µs : {} images", frame.pts_us, pieces_.size())});
  }
  std::vector<PackSize> sizes;
  sizes.reserve(pieces_.size());
  for (const ass_image* image : pieces_) sizes.push_back({.width = image->w, .height = image->h});
  auto packing = pack(sizes, kMaxAtlasSide);
  if (!packing) {
    return refuse_content({packing.error().code,
                           std::format("sous-titres à {} µs : {}", frame.pts_us, packing.error().detail)});
  }

  std::unique_ptr<SubtitleAtlas> atlas = pool.take();
  if (atlas == nullptr) return WaitForAtlas{};
  copy_pieces(pieces_, *packing, *atlas);
  atlas->generation = ++generation_;

  rects_.clear();
  for (std::size_t i = 0; i < pieces_.size(); ++i) {
    const ass_image& image = *pieces_[i];
    const Rgba8 color = color_of(image.color);
    rects_.push_back({.src = {.x = packing->positions[i].x,
                              .y = packing->positions[i].y,
                              .width = image.w,
                              .height = image.h},
                      .dst_x = image.dst_x,
                      .dst_y = image.dst_y,
                      .color = vsfilter_ ? vsfilter_bt601_on_bt709(color) : color});
  }
  current_ = pool.share(std::move(atlas));
  in_sync_ = true;
  frame.atlas = current_;
  frame.rects = rects_;
  return frame;
}

}  // namespace ajp::subs
