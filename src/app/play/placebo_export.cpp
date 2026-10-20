#include "app/play/placebo_export.h"

#include <algorithm>
#include <bit>
#include <format>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include <libplacebo/renderer.h>

#include "app/play/placebo_impl.h"
#include "core/log.h"

namespace ajp::app::play {
namespace {

// Alignement de la mémoire d'un tampon : celui des pointeurs hôte que Vulkan sait importer
// (VK_EXT_external_memory_host), pour que la relecture écrive directement dedans.
constexpr std::size_t kHostAlign = 4096;
// Pas minimal d'une ligne : celui des images CUDA de FFmpeg, que NVENC enregistre tel quel.
constexpr std::size_t kPitchAlign = 256;
// La mémoire d'un tampon d'appareil telle que CUDA l'importe : descripteur POSIX, ou HANDLE NT.
#if defined(_WIN32)
constexpr pl_handle_type kCudaHandle = PL_HANDLE_WIN32;
#else
constexpr pl_handle_type kCudaHandle = PL_HANDLE_FD;
#endif

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
// disposition (P010 : 10 bits dans les poids forts de 16 ; packed : la texture les y place
// elle-même, rx10).
pl_color_repr repr_of(exporter::Layout layout, const ColorInfo& color, bool packed) {
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
      repr.bits = packed ? pl_bit_encoding{.sample_depth = 10, .color_depth = 10, .bit_shift = 0}
                         : pl_bit_encoding{.sample_depth = 16, .color_depth = 10, .bit_shift = 6};
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

int export_cuda_ordinal(const GpuDevice& device) {
  if (device.vendor() != GpuVendor::Nvidia) return -1;
  auto driver = media::CudaDriver::load();
  if (!driver) return -1;
  auto ordinal = (*driver)->ordinal_of(device.uuid);
  return ordinal ? *ordinal : -1;
}

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
  const std::size_t offset_align = std::max<std::size_t>(1, gpu->limits.align_tex_xfer_offset);
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
    // Les plans à la suite, au même pas (la chroma entrelacée a autant d'octets par ligne que la
    // luma) : la disposition des images CUDA de FFmpeg, que NVENC lit d'un seul enregistrement.
    if (end % offset_align != 0)
      return refuse(
          RefusalCode::FeatureUnavailable,
          std::format("export : plan {} à l'offset {}, multiple de {} attendu", p, end, offset_align));
    out->offsets_[p] = end;
    out->pitches_[p] = align_up(static_cast<std::size_t>(params.w) * format->texel_size, pitch_align);
    end = out->offsets_[p] + out->pitches_[p] * static_cast<std::size_t>(params.h);
  }
  out->bytes_ = align_up(end, kHostAlign);
  out->async_ = gpu->limits.callbacks;
  const auto log = log_channel("export");
  if (setup.vulkan_frames != nullptr) {
    if (auto opened = out->open_vulkan(gpu, setup.vulkan_frames)) {
      log->info("export : images rendues dans celles de l'encodeur Vulkan Video, sur le GPU");
      return out;
    } else {
      log->warn("export : tampons hôte, l'encodeur chargera ses images : {}", opened.error().detail);
      out->close_vulkan(gpu);
    }
  }
  if (setup.cuda) {
    if (auto opened = out->open_cuda(gpu)) {
      log->info("export : images rendues en mémoire CUDA, lues en place par l'encodeur");
      return out;
    } else {
      log->warn("export : tampons hôte, l'encodeur chargera ses images : {}", opened.error().detail);
      out->close_cuda(gpu);
    }
  }
  for (Buffer& buffer : out->buffers_) {
    buffer.data =
        static_cast<std::uint8_t*>(::operator new[](out->bytes_, std::align_val_t{kHostAlign}, std::nothrow));
    if (buffer.data == nullptr)
      return refuse(RefusalCode::GpuMemoryExhausted, "export : mémoire hôte des tampons");
  }
  return out;
}

std::expected<void, Refusal> FrameExport::open_cuda(pl_gpu gpu) {
  if ((gpu->export_caps.buf & kCudaHandle) == 0)
    return refuse(RefusalCode::FeatureUnavailable, "tampons Vulkan non exportables vers CUDA");
  auto driver = media::CudaDriver::load();
  if (!driver) return std::unexpected(driver.error());
  auto ordinal = (*driver)->ordinal_of(screen_.device().uuid);
  if (!ordinal) return std::unexpected(ordinal.error());
  auto primary = (*driver)->retain_primary(*ordinal);
  if (!primary) return std::unexpected(primary.error());
  cuda_ = Cuda{.driver = *driver, .primary = *primary};
  for (Buffer& buffer : buffers_) {
    pl_buf_params params{};
    params.size = bytes_;
    params.memory_type = PL_BUF_MEM_DEVICE;
    params.export_handle = kCudaHandle;
    buffer.buf = pl_buf_create(gpu, &params);
    if (buffer.buf == nullptr) return refuse(RefusalCode::GpuMemoryExhausted, "tampon d'appareil exporté");
    const pl_shared_mem& mem = buffer.buf->shared_mem;
#if defined(_WIN32)
    const auto handle = std::bit_cast<std::uintptr_t>(mem.handle.handle);
#else
    const auto handle = static_cast<std::uintptr_t>(mem.handle.fd);
#endif
    // Allocation non dédiée : libplacebo ne réserve VkMemoryDedicatedAllocateInfo qu'aux images.
    auto memory = (*driver)->import_memory(primary->context, handle, mem.size, mem.offset, bytes_, false);
    if (!memory) return std::unexpected(memory.error());
    buffer.memory = std::move(*memory);
  }
  return {};
}

void FrameExport::close_cuda(pl_gpu gpu) {
  for (Buffer& buffer : buffers_) {
    buffer.memory.owner.reset();  // CUDA rend la mémoire avant que Vulkan ne la libère
    pl_buf_destroy(gpu, &buffer.buf);
  }
  cuda_.reset();  // le contexte primaire en dernier
}

FrameExport::~FrameExport() {
  const pl_gpu gpu = screen_.impl_->vk->gpu;
  pl_gpu_finish(gpu);  // les copies en cours écrivent dans les tampons : finies avant de les rendre
  for (Buffer& buffer : buffers_) {
    if (buffer.data != nullptr) ::operator delete[](buffer.data, std::align_val_t{kHostAlign});
  }
  close_cuda(gpu);
  close_vulkan(gpu);
  for (pl_tex& tex : planes_) pl_tex_destroy(gpu, &tex);
  pl_renderer_destroy(&renderer_);
}

bool FrameExport::has_room() const {
  if (vulkan_) return vulkan_->pending.size() < kBuffers;
  return std::ranges::any_of(buffers_, [](const Buffer& b) { return !b.in_flight && !b.held.load(); });
}

FrameExport::Converted FrameExport::render(const RingFrame& frame, std::span<const pl_tex> targets,
                                           bool packed) {
  PlaceboScreen::Impl& s = *screen_.impl_;
  const pl_gpu gpu = s.vk->gpu;
  const auto given_back = [&](std::optional<Refusal> failure) {
    return Converted{.freed = s.ring.retire(gpu, frame), .failure = std::move(failure)};
  };
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
  target.num_planes = static_cast<int>(targets.size());
  for (std::size_t p = 0; p < targets.size() && p < planes.size(); ++p) {
    pl_plane& plane = target.planes[p];
    plane.texture = targets[p];
    plane.components = planes[p].components;
    for (int c = 0; c < planes[p].components; ++c)
      plane.component_mapping[c] = planes[p].mapping[static_cast<std::size_t>(c)];
  }
  target.repr = repr_of(setup_.layout, setup_.color, packed);
  target.color = s.source;  // la couleur du ring, sans conversion : ni HDR ni grade à l'export
  target.crop = {0, 0, static_cast<float>(setup_.width), static_cast<float>(setup_.height)};
  pl_frame_set_chroma_location(&target, chroma_of(setup_.color.siting));

  // Les filtres les plus fins de libplacebo (EWA Lanczos), sans débandage : il ajouterait du grain
  // à une image que SPAN a déjà nettoyée. Tramage vers la profondeur de la sortie.
  pl_render_params params = pl_render_high_quality_params;
  params.deband_params = nullptr;
  const bool rendered = pl_render_image(renderer_, &image, &target, &params);
  Converted out = given_back(std::nullopt);  // la copie lit les textures de sortie, plus le slot
  if (!rendered) out.failure = Refusal{RefusalCode::VendorCallFailed, "export : pl_render_image"};
  return out;
}

FrameExport::Converted FrameExport::convert(const RingFrame& frame) {
  if (vulkan_) return convert_vulkan(frame);
  PlaceboScreen::Impl& s = *screen_.impl_;
  const pl_gpu gpu = s.vk->gpu;
  const auto free =
      std::ranges::find_if(buffers_, [](const Buffer& b) { return !b.in_flight && !b.held.load(); });
  if (free == buffers_.end()) {
    return Converted{.freed = s.ring.retire(gpu, frame),
                     .failure = Refusal{RefusalCode::SequenceOrder, "export : aucun tampon libre"}};
  }
  Converted out =
      render(frame, std::span<const pl_tex>(planes_.data(), static_cast<std::size_t>(plane_count_)), false);
  if (out.failure) return out;
  const std::vector<PlaneFormat> planes = planes_of(setup_.layout);

  Buffer& buffer = *free;
  buffer.planes_done = 0;
  buffer.in_flight = true;
  buffer.pts_us = frame.pts_us;
  const auto failed = [&](const char* what) {
    pl_gpu_finish(gpu);  // les copies déjà lancées écrivent encore dans le tampon
    buffer.in_flight = false;
    out.failure = Refusal{RefusalCode::VendorCallFailed, std::format("export : {}", what)};
    return out;
  };
  for (std::size_t p = 0; p < planes.size(); ++p) {
    pl_tex_transfer_params download{};
    download.tex = planes_[p];
    download.row_pitch = pitches_[p];
    if (buffer.buf != nullptr) {
      download.buf = buffer.buf;  // copie sur le GPU ; prêt quand le tampon n'est plus en usage
      download.buf_offset = offsets_[p];
    } else {
      download.ptr = buffer.data + offsets_[p];
      if (async_) {
        download.callback = [](void* priv) { ++static_cast<Buffer*>(priv)->planes_done; };
        download.priv = &buffer;
      }
    }
    if (!pl_tex_download(gpu, &download)) return failed("pl_tex_download");
  }
  // Le tampon passe à CUDA (barrière vers la file externe) jusqu'à sa prochaine copie, qui le
  // reprend ; l'encodeur ne le lit qu'une fois ready(), et le rend avant qu'il ne soit réutilisé.
  if (buffer.buf != nullptr && !pl_buf_export(gpu, buffer.buf)) return failed("pl_buf_export");
  if (!async_ && buffer.buf == nullptr) buffer.planes_done = plane_count_;
  order_.push_back(static_cast<std::size_t>(free - buffers_.begin()));
  pl_gpu_flush(gpu);
  return out;
}

bool FrameExport::ready(Buffer& buffer) {
  const pl_gpu gpu = screen_.impl_->vk->gpu;
  if (buffer.buf != nullptr) return !pl_buf_poll(gpu, buffer.buf, 0);  // copie et export finis
  if (buffer.planes_done >= plane_count_) return true;
  // Les rappels des relectures finies passent pendant une interrogation de libplacebo.
  static_cast<void>(pl_tex_poll(gpu, planes_[0], 0));
  return buffer.planes_done >= plane_count_;
}

bool FrameExport::deliver(const std::function<bool(exporter::Picture&&)>& push) {
  if (vulkan_) return deliver_vulkan(push);
  if (order_.empty()) return false;
  Buffer& buffer = buffers_[order_.front()];
  if (!ready(buffer)) return false;
  exporter::Picture picture;
  picture.device = buffer.buf != nullptr;
  for (int p = 0; p < plane_count_; ++p) {
    const auto i = static_cast<std::size_t>(p);
    // En mémoire CUDA, l'adresse d'appareil tient lieu de pointeur : FFmpeg ne la déréférence pas.
    picture.planes[i] = picture.device ? reinterpret_cast<std::uint8_t*>(buffer.memory.pointer + offsets_[i])
                                       : buffer.data + offsets_[i];
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
