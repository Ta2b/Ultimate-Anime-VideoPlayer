#include "app/play/placebo_export.h"

#include <algorithm>
#include <format>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include <libplacebo/renderer.h>

#include "app/play/placebo_impl.h"

namespace ajp::app::play {
namespace {

// Alignement de la mémoire d'un tampon : celui des pointeurs hôte que Vulkan sait importer
// (VK_EXT_external_memory_host), pour que la relecture écrive directement dedans.
constexpr std::size_t kHostAlign = 4096;
constexpr std::size_t kPitchAlign = 256;

std::size_t align_up(std::size_t value, std::size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

// Un plan de sortie : son format de texture, ses composantes (Y = 0, Cb = 1, Cr = 2) et son
// sous-échantillonnage.
struct PlaneFormat {
  const char* format;
  int components;
  std::array<int, 2> mapping;
  int divisor;  // 2 pour la chroma du 4:2:0
};

std::vector<PlaneFormat> planes_of(exporter::Layout layout) {
  switch (layout) {
    case exporter::Layout::P010:
      return {{"r16", 1, {0, 0}, 1}, {"rg16", 2, {1, 2}, 2}};
    case exporter::Layout::Nv12:
      return {{"r8", 1, {0, 0}, 1}, {"rg8", 2, {1, 2}, 2}};
    case exporter::Layout::Yuv420p10:
      return {{"r16", 1, {0, 0}, 1}, {"r16", 1, {1, 0}, 2}, {"r16", 1, {2, 0}, 2}};
  }
  return {};
}

// Le codage YUV de la sortie : matrice et niveaux de la couleur des images, profondeur de la
// disposition (P010 : 10 bits dans les poids forts de 16).
pl_color_repr repr_of(exporter::Layout layout, const ColorInfo& color) {
  pl_color_repr repr{};
  switch (color.matrix) {
    case ColorMatrix::Bt601:
      repr.sys = PL_COLOR_SYSTEM_BT_601;
      break;
    case ColorMatrix::Bt2020Ncl:
      repr.sys = PL_COLOR_SYSTEM_BT_2020_NC;
      break;
    case ColorMatrix::Bt709:
    case ColorMatrix::Rgb:
      repr.sys = PL_COLOR_SYSTEM_BT_709;
      break;
  }
  repr.levels = color.range == ColorRange::Full ? PL_COLOR_LEVELS_FULL : PL_COLOR_LEVELS_LIMITED;
  repr.alpha = PL_ALPHA_NONE;
  switch (layout) {
    case exporter::Layout::P010:
      repr.bits = {.sample_depth = 16, .color_depth = 10, .bit_shift = 6};
      break;
    case exporter::Layout::Yuv420p10:
      repr.bits = {.sample_depth = 16, .color_depth = 10, .bit_shift = 0};
      break;
    case exporter::Layout::Nv12:
      repr.bits = {.sample_depth = 8, .color_depth = 8, .bit_shift = 0};
      break;
  }
  return repr;
}

pl_chroma_location chroma_of(ChromaSiting siting) {
  switch (siting) {
    case ChromaSiting::Left:
      return PL_CHROMA_LEFT;
    case ChromaSiting::Center:
      return PL_CHROMA_CENTER;
    case ChromaSiting::TopLeft:
      return PL_CHROMA_TOP_LEFT;
  }
  return PL_CHROMA_LEFT;
}

}  // namespace

FrameExport::FrameExport(PlaceboScreen& screen, const FrameExportSetup& setup)
    : screen_(screen), setup_(setup) {}

std::expected<std::unique_ptr<FrameExport>, Refusal> FrameExport::open(PlaceboScreen& screen,
                                                                       const FrameExportSetup& setup) {
  std::unique_ptr<FrameExport> out(new FrameExport(screen, setup));
  const PlaceboScreen::Impl& s = *screen.impl_;
  const pl_gpu gpu = s.vk->gpu;
  out->renderer_ = pl_renderer_create(s.log, gpu);
  if (out->renderer_ == nullptr) return refuse(RefusalCode::VendorCallFailed, "export : pl_renderer_create");
  const std::vector<PlaneFormat> planes = planes_of(setup.layout);
  out->plane_count_ = static_cast<int>(planes.size());
  const std::size_t pitch_align = std::max(kPitchAlign, gpu->limits.align_tex_xfer_pitch);
  std::size_t end = 0;
  for (std::size_t p = 0; p < planes.size(); ++p) {
    const PlaneFormat& plane = planes[p];
    const pl_fmt format = pl_find_named_fmt(gpu, plane.format);
    constexpr auto kNeeded = static_cast<pl_fmt_caps>(PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_HOST_READABLE);
    if (format == nullptr || (format->caps & kNeeded) != kNeeded)
      return refuse(RefusalCode::FeatureUnavailable,
                    std::format("export : texture {} sans rendu ni relecture sur ce GPU", plane.format));
    pl_tex_params params{};
    params.w = setup.width / plane.divisor;
    params.h = setup.height / plane.divisor;
    params.format = format;
    params.renderable = true;
    params.host_readable = true;
    out->planes_[p] = pl_tex_create(gpu, &params);
    if (out->planes_[p] == nullptr) return refuse(RefusalCode::VendorCallFailed, "export : pl_tex_create");
    out->offsets_[p] = align_up(end, kHostAlign);
    out->pitches_[p] = align_up(static_cast<std::size_t>(params.w) * format->texel_size, pitch_align);
    end = out->offsets_[p] + out->pitches_[p] * static_cast<std::size_t>(params.h);
  }
  out->bytes_ = align_up(end, kHostAlign);
  out->async_ = gpu->limits.callbacks;
  for (Buffer& buffer : out->buffers_) {
    buffer.data =
        static_cast<std::uint8_t*>(::operator new[](out->bytes_, std::align_val_t{kHostAlign}, std::nothrow));
    if (buffer.data == nullptr)
      return refuse(RefusalCode::GpuMemoryExhausted, "export : mémoire hôte des tampons");
  }
  return out;
}

FrameExport::~FrameExport() {
  const pl_gpu gpu = screen_.impl_->vk->gpu;
  pl_gpu_finish(gpu);  // les relectures en cours écrivent dans les tampons : finies avant de les rendre
  for (Buffer& buffer : buffers_) {
    if (buffer.data != nullptr) ::operator delete[](buffer.data, std::align_val_t{kHostAlign});
  }
  for (pl_tex& tex : planes_) pl_tex_destroy(gpu, &tex);
  pl_renderer_destroy(&renderer_);
}

bool FrameExport::has_room() const {
  return std::ranges::any_of(buffers_, [](const Buffer& b) { return !b.in_flight && !b.held.load(); });
}

FrameExport::Converted FrameExport::convert(const RingFrame& frame) {
  PlaceboScreen::Impl& s = *screen_.impl_;
  const pl_gpu gpu = s.vk->gpu;
  const auto given_back = [&](std::optional<Refusal> failure) {
    return Converted{.freed = s.ring.retire(gpu, frame), .failure = std::move(failure)};
  };
  const auto free =
      std::ranges::find_if(buffers_, [](const Buffer& b) { return !b.in_flight && !b.held.load(); });
  if (free == buffers_.end())
    return given_back(Refusal{RefusalCode::SequenceOrder, "export : aucun tampon libre"});
  auto tex = s.ring.acquire(gpu, frame);
  if (!tex) return given_back(std::move(tex.error()));

  pl_frame image{};
  image.num_planes = 1;
  image.planes[0].texture = *tex;
  image.planes[0].components = 3;
  for (int c = 0; c < 3; ++c) image.planes[0].component_mapping[c] = c;
  image.repr = pl_color_repr_rgb;
  image.color = s.source;
  image.crop = {0, 0, static_cast<float>((*tex)->params.w), static_cast<float>((*tex)->params.h)};

  pl_frame target{};
  const std::vector<PlaneFormat> planes = planes_of(setup_.layout);
  target.num_planes = plane_count_;
  for (std::size_t p = 0; p < planes.size(); ++p) {
    pl_plane& plane = target.planes[p];
    plane.texture = planes_[p];
    plane.components = planes[p].components;
    for (int c = 0; c < planes[p].components; ++c)
      plane.component_mapping[c] = planes[p].mapping[static_cast<std::size_t>(c)];
  }
  target.repr = repr_of(setup_.layout, setup_.color);
  target.color = s.source;  // la couleur du ring, sans conversion : ni HDR ni grade à l'export
  target.crop = {0, 0, static_cast<float>(setup_.width), static_cast<float>(setup_.height)};
  pl_frame_set_chroma_location(&target, chroma_of(setup_.color.siting));

  // Les filtres les plus fins de libplacebo (EWA Lanczos), sans débandage : il ajouterait du grain
  // à une image que SPAN a déjà nettoyée. Tramage vers la profondeur de la sortie.
  pl_render_params params = pl_render_high_quality_params;
  params.deband_params = nullptr;
  const bool rendered = pl_render_image(renderer_, &image, &target, &params);
  Converted out = given_back(std::nullopt);  // la relecture lit les textures de sortie, plus le slot
  if (!rendered) {
    out.failure = Refusal{RefusalCode::VendorCallFailed, "export : pl_render_image"};
    return out;
  }

  Buffer& buffer = *free;
  buffer.planes_done = 0;
  buffer.in_flight = true;
  buffer.pts_us = frame.pts_us;
  for (std::size_t p = 0; p < planes.size(); ++p) {
    pl_tex_transfer_params download{};
    download.tex = planes_[p];
    download.row_pitch = pitches_[p];
    download.ptr = buffer.data + offsets_[p];
    if (async_) {
      download.callback = [](void* priv) { ++static_cast<Buffer*>(priv)->planes_done; };
      download.priv = &buffer;
    }
    if (!pl_tex_download(gpu, &download)) {
      pl_gpu_finish(gpu);  // les relectures déjà lancées écrivent encore dans le tampon
      buffer.in_flight = false;
      out.failure = Refusal{RefusalCode::VendorCallFailed, "export : pl_tex_download"};
      return out;
    }
  }
  if (!async_) buffer.planes_done = plane_count_;
  order_.push_back(static_cast<std::size_t>(free - buffers_.begin()));
  pl_gpu_flush(gpu);
  return out;
}

bool FrameExport::ready(Buffer& buffer) {
  if (buffer.planes_done >= plane_count_) return true;
  // Les rappels des relectures finies passent pendant une interrogation de libplacebo.
  static_cast<void>(pl_tex_poll(screen_.impl_->vk->gpu, planes_[0], 0));
  return buffer.planes_done >= plane_count_;
}

bool FrameExport::deliver(const std::function<bool(exporter::Picture&&)>& push) {
  if (order_.empty()) return false;
  Buffer& buffer = buffers_[order_.front()];
  if (!ready(buffer)) return false;
  exporter::Picture picture;
  for (int p = 0; p < plane_count_; ++p) {
    const auto i = static_cast<std::size_t>(p);
    picture.planes[i] = buffer.data + offsets_[i];
    picture.pitches[i] = static_cast<int>(pitches_[i]);
  }
  picture.bytes = bytes_;
  picture.pts_us = buffer.pts_us;
  picture.release = [&buffer] { buffer.held.store(false); };
  buffer.held.store(true);
  if (!push(std::move(picture))) {
    buffer.held.store(false);
    return false;
  }
  buffer.in_flight = false;
  order_.pop_front();
  return true;
}

}  // namespace ajp::app::play
