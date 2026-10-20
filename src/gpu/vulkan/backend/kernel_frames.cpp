// Images décodées prêtes à lire : les plans de la VkImage multiplan de FFmpeg lus en place
// (échantillonnable à format mutable : rien à copier), sinon un tampon YUV de disposition serrée,
// rempli depuis la mémoire hôte (décodage logiciel, tests) par le fil d'envoi, ou par le GPU
// depuis l'image, entre begin_vulkan_access et end_vulkan_access ; une image déjà copiée se
// relit depuis son tampon (mémo). Puis l'ingestion.
#include <algorithm>
#include <bit>
#include <cstring>
#include <format>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

#include "core/thread.h"
#include "gpu/vulkan/backend/kernel_ops.h"
#include "gpu/vulkan/backend/kernel_ops_detail.h"
#include "gpu/vulkan/backend/reduce_weights.h"
#include "kernels/spec/color.h"
#include "kernels/spec/rife.h"

namespace ajp::gpu::vk {

namespace detail {

std::expected<kernels::Format, Refusal> kernel_format(PixelFormat format) {
  switch (format) {
    case PixelFormat::Nv12:
      return kernels::Format::Nv12;
    case PixelFormat::P010:
      return kernels::Format::P010;
    case PixelFormat::P016:
      return kernels::Format::P016;
    case PixelFormat::Yuv420p:
      return kernels::Format::Yuv420p;
    case PixelFormat::Yuv420p10:
      return kernels::Format::Yuv420p10;
    case PixelFormat::Yuv444p16:
      return kernels::Format::Yuv444p16;
    default:
      return std::unexpected(
          Refusal{RefusalCode::FormatUnsupported, "image décodée : format RGB, YUV attendu"});
  }
}

std::expected<ColorSpec, Refusal> color_spec(const ColorInfo& color) {
  ColorSpec spec;
  switch (color.matrix) {
    case ColorMatrix::Bt601:
      spec.matrix = kernels::Matrix::Bt601;
      break;
    case ColorMatrix::Bt709:
      spec.matrix = kernels::Matrix::Bt709;
      break;
    case ColorMatrix::Bt2020Ncl:
      spec.matrix = kernels::Matrix::Bt2020Ncl;
      break;
    case ColorMatrix::Rgb:
      return std::unexpected(
          Refusal{RefusalCode::FormatUnsupported, "image décodée : matrice RGB sur du YUV"});
  }
  spec.range = color.range == ColorRange::Full ? kernels::Range::Full : kernels::Range::Limited;
  spec.siting = color.siting == ChromaSiting::Center    ? kernels::Siting::Center
                : color.siting == ChromaSiting::TopLeft ? kernels::Siting::TopLeft
                                                        : kernels::Siting::Left;
  return spec;
}

VkImage to_image(std::uintptr_t handle) {
  return from_bits<VkImage>(handle);
}

VkSemaphore to_semaphore(std::uintptr_t handle) {
  return from_bits<VkSemaphore>(handle);
}

std::uint64_t image_bits(VkImage image) {
  return to_bits(image);
}

std::uint32_t plane_copies(const YuvView& v, std::array<PlaneCopy, 3>& out) {
  const kernels::FormatInfo info = kernels::info(v.format);
  const auto bytes = static_cast<std::uint32_t>(info.bytes);
  const std::uint32_t cw = chroma_width(v.format, v.width);
  const std::uint32_t ch = chroma_height(v.format, v.height);
  out[0] = {.width = v.width, .height = v.height, .texel = bytes, .dst = v.layout.luma};
  if (info.interleaved) {
    out[1] = {.width = cw, .height = ch, .texel = 2 * bytes, .dst = v.layout.cb};
    return 2;
  }
  out[1] = {.width = cw, .height = ch, .texel = bytes, .dst = v.layout.cb};
  out[2] = {.width = cw, .height = ch, .texel = bytes, .dst = v.layout.cr};
  return 3;
}

}  // namespace detail

namespace {

using detail::add_view;
using detail::handle_bits;
using detail::image_bits;
using detail::plane_copies;
using detail::PlaneCopy;
using detail::tensor_view;
using detail::to_semaphore;

std::unexpected<Refusal> refuse_frame(RefusalCode code, std::string text) {
  return std::unexpected(Refusal{.code = code, .detail = std::move(text)});
}

}  // namespace

std::expected<std::size_t, Refusal> VulkanKernelOps::staging_for(kernels::Format format, std::uint32_t width,
                                                                 std::uint32_t height, MemoryUse use,
                                                                 std::size_t avoid) {
  const auto completed = queue_.completed();
  if (!completed) return std::unexpected(completed.error());
  const std::uint64_t uploaded = uploads_done();
  ++staging_uses_;
  std::size_t same = 0;
  std::size_t oldest = SIZE_MAX;
  std::size_t held_free = SIZE_MAX;  // libre, mais son image vit encore
  std::size_t spare = SIZE_MAX;      // libre, d'une autre forme et sans service depuis longtemps
  // Libre : la file a fini de le lire, et le fil d'envoi d'y écrire.
  const auto free = [&](const Staging& s) { return s.busy_until <= *completed && s.upload <= uploaded; };
  for (std::size_t k = 0; k < staging_.size(); ++k) {
    Staging& s = staging_[k];
    if (k == avoid) continue;
    if (s.format != format || s.width != width || s.height != height || s.buffer.use() != use) {
      if (spare == SIZE_MAX && free(s) && staging_uses_ - s.used > kStagingIdleUses) spare = k;
      continue;
    }
    ++same;
    if (free(s)) {
      // Un tampon libre dont l'image vit encore garde son mémo pour une relecture : un autre
      // libre passe d'abord.
      if (s.held_ptr == nullptr || s.held.expired()) {
        s.used = staging_uses_;
        return k;
      }
      if (held_free == SIZE_MAX) held_free = k;
      continue;
    }
    if (oldest == SIZE_MAX || s.busy_until < staging_[oldest].busy_until) oldest = k;
  }
  if (held_free != SIZE_MAX) {
    staging_[held_free].used = staging_uses_;
    return held_free;
  }
  if (same >= kStagingSlots && oldest != SIZE_MAX) {
    // Tous occupés : le plus ancien se libère d'abord (attente hôte comptée, rare), et son envoi
    // s'il en reste un.
    if (auto ok = context_.wait_on_host(queue_.at(staging_[oldest].busy_until)); !ok) {
      return std::unexpected(ok.error());
    }
    if (auto ok = uploads_wait(); !ok) return std::unexpected(ok.error());
    staging_[oldest].used = staging_uses_;
    return oldest;
  }
  auto layout = tight_layout(format, width, height);
  if (!layout) return detail::contract_refusal(layout.error());
  auto buffer = Buffer::create(context_, memory_, layout->bytes, use);
  if (!buffer) return std::unexpected(buffer.error());
  Staging made{.format = format,
               .width = width,
               .height = height,
               .buffer = std::move(*buffer),
               .layout = *layout,
               .used = staging_uses_};
  if (spare != SIZE_MAX) {
    // Un tampon libre d'une autre forme (fichier précédent, autre taille) cède sa place : les
    // tampons de transfert ne s'accumulent pas d'un fichier à l'autre. Sa destruction entre au
    // journal, qui fait oublier les enregistrements qui le liaient.
    staging_[spare] = std::move(made);
    return spare;
  }
  staging_.push_back(std::move(made));
  return staging_.size() - 1;
}

std::optional<std::size_t> VulkanKernelOps::memo_for(const DecodedFrame& frame, kernels::Format format,
                                                     std::uint32_t width, std::uint32_t height,
                                                     std::size_t avoid) const {
  for (std::size_t k = 0; k < staging_.size(); ++k) {
    const Staging& s = staging_[k];
    if (k == avoid || s.held_ptr != frame.data.get() || s.held.expired()) continue;
    if (s.format == format && s.width == width && s.height == height) return k;
  }
  return std::nullopt;
}

std::expected<VulkanKernelOps::Source, Refusal> VulkanKernelOps::source(const DecodedFrame& frame,
                                                                        std::size_t avoid, bool direct) {
  if (frame.data == nullptr) return refuse_frame(RefusalCode::ShapeInvalid, "image décodée sans données");
  auto format = detail::kernel_format(frame.format);
  if (!format) return std::unexpected(format.error());
  if (!kernels::valid_frame_dimension(frame.width) || !kernels::valid_frame_dimension(frame.height)) {
    return refuse_frame(RefusalCode::ShapeInvalid,
                        std::format("image décodée {}x{} : dimensions paires dans [2, {}] attendues",
                                    frame.width, frame.height, kernels::kMaxDimension));
  }
  const auto w = static_cast<std::uint32_t>(frame.width);
  const auto h = static_cast<std::uint32_t>(frame.height);
  const FrameMemory memory = frame.data->memory();
  if (memory != FrameMemory::Host && memory != FrameMemory::Vulkan) {
    return refuse_frame(RefusalCode::FormatUnsupported, "image décodée hors de la mémoire hôte ou Vulkan");
  }
  Source src;
  if (memory == FrameMemory::Vulkan) {
    src.image = detail::to_image(frame.data->native_plane(0));
    if (src.image == VK_NULL_HANDLE)
      return refuse_frame(RefusalCode::ShapeInvalid, "image Vulkan sans VkImage");
    if (const std::uintptr_t pool = frame.data->native_pool_id(); pool != pool_id_) {
      // Nouveau fichier, flux qui change de taille : l'ancien lot est rendu, ses images peuvent
      // disparaître et leurs poignées revenir dans le nouveau ; ses vues partent avant lui.
      release_plane_views();
      pool_ = frame.data->native_pool();
      pool_id_ = pool;
      ++pool_generation_;
    }
    src.pool = pool_generation_;
  }
  if (const std::optional<std::size_t> memo = memo_for(frame, *format, w, h, avoid)) {
    // L'image est déjà dans ce tampon, que la file lit dans l'ordre : ni copie ni accès.
    const Staging& s = staging_[*memo];
    staging_[*memo].used = ++staging_uses_;
    src.view = {.buffer = s.buffer.view(), .format = *format, .width = w, .height = h, .layout = s.layout};
    src.staging = *memo;
    src.image = VK_NULL_HANDLE;
    src.pool = 0;
    return src;
  }
  if (memory == FrameMemory::Vulkan) {
    auto access = frame.data->begin_vulkan_access();
    if (!access) return std::unexpected(access.error());
    src.access = *access;
    if (direct && reads_in_place(*access, *format)) {
      auto planes = plane_views_for(src.image, *format, w, h);
      if (planes) {
        src.planes = *planes;
        src.staging = SIZE_MAX;
        return src;
      }
      // Les vues ne se créent pas (rare : appareil à court) : la copie prend le relais.
      frame.data->end_vulkan_access(access->state, false);
      return std::unexpected(planes.error());
    }
  }
  const MemoryUse use = memory == FrameMemory::Host ? MemoryUse::Upload : MemoryUse::Device;
  auto slot = staging_for(*format, w, h, use, avoid);
  if (!slot) {
    if (src.access) frame.data->end_vulkan_access(src.access->state, false);
    return std::unexpected(slot.error());
  }
  Staging& s = staging_[*slot];
  s.held_ptr = nullptr;  // le tampon change d'image : le mémo tombe jusqu'à la soumission
  s.held.reset();
  src.view = {.buffer = s.buffer.view(), .format = *format, .width = w, .height = h, .layout = s.layout};
  src.staging = *slot;
  if (memory == FrameMemory::Vulkan) return src;
  std::array<PlaneCopy, 3> planes{};
  const std::uint32_t count = plane_copies(src.view, planes);
  if (frame.data->plane_count() != static_cast<int>(count)) {
    return refuse_frame(RefusalCode::ShapeInvalid, "image hôte : nombre de plans inattendu");
  }
  std::array<UploadPlane, 3> copies{};
  for (std::uint32_t p = 0; p < count; ++p) {
    const PlaneCopy& pc = planes[p];
    const std::span<const std::byte> in = frame.data->host_plane(static_cast<int>(p));
    const std::size_t stride = frame.data->host_stride(static_cast<int>(p));
    const std::size_t row = std::size_t{pc.width} * pc.texel;
    if (stride < row || in.size() < (stride * (pc.height - 1)) + row) {
      return refuse_frame(RefusalCode::ShapeInvalid, std::format("image hôte : plan {} trop court", p));
    }
    copies[p] = {.in = in.data(),
                 .stride = stride,
                 .out = s.buffer.mapped() + pc.dst.offset,
                 .pitch = pc.dst.pitch,
                 .row = static_cast<std::uint32_t>(row),
                 .rows = pc.height};
  }
  // La copie se fait sur le fil d'envoi pendant que ce fil enregistre la suite ; le lot
  // l'attend avant de partir (uploads_wait).
  if (auto ok = upload(frame, *slot, std::span(copies).first(count)); !ok) return std::unexpected(ok.error());
  return src;
}

std::expected<void, Refusal> VulkanKernelOps::upload(const DecodedFrame& frame, std::size_t slot,
                                                     std::span<const UploadPlane> planes) {
  if (!upload_thread_.joinable()) {
    auto thread = start_thread(
        "envoi", ThreadReport([this](const Refusal& refusal) {
          const std::scoped_lock lock(uploads_mutex_);
          if (!upload_failure_) upload_failure_ = refusal;
          uploads_finished_.notify_all();
        }),
        [this](const std::stop_token& stop, const ThreadReport& /*report*/) { serve_uploads(stop); });
    if (!thread) return std::unexpected(std::move(thread.error()));
    upload_thread_ = std::move(*thread);
  }
  const auto job = [&] {
    Upload made{.frame = frame.data, .planes = {}, .count = static_cast<std::uint32_t>(planes.size())};
    std::ranges::copy(planes, made.planes.begin());
    return made;
  };
  if (!uploads_.try_push(job())) {
    // Boîte pleine : les envois en file sont attendus (le fil du GPU est en avance), puis le
    // nôtre entre.
    if (auto ok = uploads_wait(); !ok) return ok;
    if (!uploads_.try_push(job()))
      return refuse_frame(RefusalCode::ThreadUnavailable, "fil d'envoi : boîte pleine après l'attente");
  }
  const std::scoped_lock lock(uploads_mutex_);
  if (upload_failure_) return std::unexpected(*upload_failure_);
  staging_[slot].upload = ++uploads_queued_;
  return {};
}

void VulkanKernelOps::serve_uploads(const std::stop_token& stop) {
  // Chaque envoi en file est fait, même à l'arrêt : le lot l'attend peut-être.
  while (std::optional<Inbox<Upload>::Message> message = uploads_.wait(stop)) {
    std::optional<Upload> job(std::get<Upload>(std::move(*message)));
    for (std::uint32_t p = 0; p < job->count; ++p) {
      const UploadPlane& plane = job->planes[p];
      for (std::uint32_t y = 0; y < plane.rows; ++y) {
        std::memcpy(plane.out + (std::size_t{y} * plane.pitch), plane.in + (std::size_t{y} * plane.stride),
                    plane.row);
      }
    }
    job.reset();  // l'image est rendue avant d'annoncer l'envoi fait
    {
      const std::scoped_lock lock(uploads_mutex_);
      ++uploads_done_;
    }
    uploads_finished_.notify_all();
  }
}

std::expected<void, Refusal> VulkanKernelOps::uploads_wait() {
  std::unique_lock lock(uploads_mutex_);
  uploads_finished_.wait(lock, [&] { return uploads_done_ >= uploads_queued_ || upload_failure_; });
  if (upload_failure_) return std::unexpected(*upload_failure_);
  return {};
}

std::uint64_t VulkanKernelOps::uploads_done() const {
  const std::scoped_lock lock(uploads_mutex_);
  return uploads_done_;
}

std::uint64_t VulkanKernelOps::uploads_queued() const {
  const std::scoped_lock lock(uploads_mutex_);
  return uploads_queued_;
}

std::size_t VulkanKernelOps::uploads_pending() const {
  const std::scoped_lock lock(uploads_mutex_);
  return static_cast<std::size_t>(uploads_queued_ - uploads_done_);
}

std::expected<void, Refusal> VulkanKernelOps::submit_frames(Recording& recording, std::span<Source> sources,
                                                            std::span<const DecodedFrame* const> frames) {
  detail::AccessGuard guard;
  waits_.clear();
  signals_.clear();
  VulkanImageState after{.layout = VK_IMAGE_LAYOUT_GENERAL};
  for (std::size_t k = 0; k < sources.size(); ++k) {
    const std::optional<VulkanImageAccess>& access = sources[k].access;
    if (!access) continue;
    const VulkanImageAccess& a = *access;
    guard.hold(frames[k]->data.get(), a.state);
    waits_.push_back({.semaphore = to_semaphore(a.semaphore), .value = a.wait_value});
    signals_.push_back({.semaphore = to_semaphore(a.semaphore), .value = a.signal_value});
    if (a.state.queue_family != VK_QUEUE_FAMILY_IGNORED) after.queue_family = context_.queue_family();
  }
  auto ok = queue_.submit(recording, waits_, signals_);
  guard.release_all(after, ok.has_value());
  if (!ok) return ok;
  const std::uint64_t value = queue_.last().value;
  for (std::size_t k = 0; k < sources.size(); ++k) {
    retained_.push_back({.frame = frames[k]->data, .until = value});
    if (sources[k].staging == SIZE_MAX) continue;  // lue en place
    Staging& s = staging_[sources[k].staging];
    s.busy_until = value;
    s.held_ptr = frames[k]->data.get();  // la copie est en route : la prochaine lecture la relit
    s.held = frames[k]->data;
  }
  reclaim();
  return {};
}

void VulkanKernelOps::add_source(OpKey& key, const Source& src) {
  key.add(handle_bits(src.view.buffer.buffer)).add(image_bits(src.image)).add(src.pool);
  key.add(src.access ? src.access->state.layout : 0).add(src.access ? src.access->state.queue_family : 0);
  key.add(src.planes ? 1 : 0);  // les vues des plans suivent (image, lot)
}

std::expected<void, Refusal> VulkanKernelOps::ingest(const DecodedFrame& frame, const IngestSpec& spec,
                                                     const Tensor& rgb) {
  auto dst = tensor_view(rgb, 3, "ingestion : tenseur RGB");
  if (!dst) return std::unexpected(dst.error());
  auto color = detail::color_spec(frame.color);
  if (!color) return std::unexpected(color.error());
  kernels::Filter chroma = spec.chroma;
  if (spec.rife_conversion) {
    *color = {.matrix = kernels::kRifeMatrix, .range = color->range, .siting = kernels::kRifeSiting};
    chroma = kernels::kRifeChromaFilter;
  }
  // Tenseur plus petit que l'image : réduction fusionnée, l'image entière passe par un tenseur
  // F32 de l'enregistrement puis par les tables de reduce_weights.
  const bool reduced =
      std::cmp_not_equal(dst->width, frame.width) || std::cmp_not_equal(dst->height, frame.height);
  if (std::cmp_greater(dst->width, frame.width) || std::cmp_greater(dst->height, frame.height)) {
    return refuse_frame(RefusalCode::ShapeInvalid,
                        std::format("ingestion : image {}x{}, tenseur {}x{} plus grand", frame.width,
                                    frame.height, dst->width, dst->height));
  }
  std::optional<WeightTable> wx;
  std::optional<WeightTable> wy;
  if (reduced) {
    auto x = reduction_weights(frame.width, static_cast<int>(dst->width), spec.reduce);
    if (!x) return std::unexpected(x.error());
    auto y = reduction_weights(frame.height, static_cast<int>(dst->height), spec.reduce);
    if (!y) return std::unexpected(y.error());
    wx = std::move(*x);
    wy = std::move(*y);
  }
  auto src = source(frame);
  if (!src) return std::unexpected(src.error());
  detail::AccessGuard pending;
  if (src->access) pending.hold(frame.data.get(), src->access->state);
  OpKey key;
  key.add(10);
  add_source(key, *src);
  add_view(key, *dst);
  key.add(static_cast<std::uint64_t>(color->matrix)).add(static_cast<std::uint64_t>(color->range));
  key.add(static_cast<std::uint64_t>(color->siting)).add(static_cast<std::uint64_t>(chroma));
  if (reduced) {
    key.add(static_cast<std::uint64_t>(spec.reduce.kind));
    key.add(std::bit_cast<std::uint32_t>(spec.reduce.b)).add(std::bit_cast<std::uint32_t>(spec.reduce.c));
  }
  auto rec = recording(key, [&](Recorder& r) -> Done {
    if (auto ok = prepare_image(r, *src); !ok) return ok;
    const auto ingest = [&](const TensorView& out) {
      return src->planes ? runner_.ingest(r, *src->planes, *color, out, chroma)
                         : runner_.ingest(r, src->view, *color, out, chroma);
    };
    if (!reduced) return ingest(*dst);
    const auto w = static_cast<std::uint32_t>(frame.width);
    const auto h = static_cast<std::uint32_t>(frame.height);
    auto full = r.scratch(tensor_bytes(w, h, 3, Precision::F32));
    if (!full) return std::unexpected(full.error());
    const TensorView image{
        .buffer = *full, .width = w, .height = h, .planes = 3, .precision = Precision::F32};
    if (auto ok = ingest(image); !ok) return ok;
    return runner_.resize(r, image, *dst, *wx, *wy);
  });
  if (!rec) return std::unexpected(rec.error());
  pending.dismiss();  // submit_frames ferme l'accès, soumission faite ou non
  const std::array frames{&frame};
  std::array sources{*src};
  return submit_frames(**rec, sources, frames);
}

}  // namespace ajp::gpu::vk
