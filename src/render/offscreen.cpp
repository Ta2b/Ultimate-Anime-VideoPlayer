#include "render/offscreen.h"

#include <array>
#include <cstddef>
#include <cstring>
#include <format>
#include <utility>
#include <vector>

#include <libplacebo/renderer.h>

#include "render/c_structs.h"
#include "render/color.h"
#include "render/render_device.h"

namespace ajp::render {
namespace {

constexpr std::size_t kChannels = 4;

}  // namespace

struct OffscreenTarget::Impl {
  enum class Readback : std::uint8_t { None, Pending, Ready };

  pl_gpu gpu = nullptr;
  pl_tex tex = nullptr;
  pl_buf buf = nullptr;
  pl_frame frame{};
  Readback state = Readback::None;
  // Sans mapping persistant (rare), les octets passent par pl_buf_read dans ce tampon, alloué
  // une fois.
  std::vector<std::uint16_t> host_copy;

  Impl() = default;
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;
  ~Impl() {
    if (gpu == nullptr) return;
    pl_buf_destroy(gpu, &buf);
    pl_tex_destroy(gpu, &tex);
  }

  std::size_t values() const {
    return static_cast<std::size_t>(tex->params.w) * static_cast<std::size_t>(tex->params.h) * kChannels;
  }
};

std::expected<OffscreenTarget, Refusal> OffscreenTarget::create(const RenderDevice& device,
                                                                const OffscreenOptions& options) {
  const pl_gpu gpu = device.gpu();
  if (options.width <= 0 || options.height <= 0 ||
      static_cast<std::uint32_t>(options.width) > gpu->limits.max_tex_2d_dim ||
      static_cast<std::uint32_t>(options.height) > gpu->limits.max_tex_2d_dim) {
    return std::unexpected(Refusal{RefusalCode::FieldOutOfRange,
                                   std::format("cible hors écran {}x{}", options.width, options.height)});
  }
  if (auto ok = core::check_display(options.display); !ok) return std::unexpected(ok.error());
  const pl_fmt format = pl_find_named_fmt(gpu, "rgba16");
  const unsigned needed =
      detail::flag_bits(PL_FMT_CAP_RENDERABLE, PL_FMT_CAP_BLENDABLE, PL_FMT_CAP_HOST_READABLE);
  if (format == nullptr || !detail::has_all(format->caps, needed)) {
    return std::unexpected(
        Refusal{RefusalCode::FeatureUnavailable, "rgba16 rendable, mélangeable et lisible"});
  }

  auto impl = std::make_unique<Impl>();
  impl->gpu = gpu;
  auto tex_params = detail::zeroed<pl_tex_params>();
  tex_params.w = options.width;
  tex_params.h = options.height;
  tex_params.format = format;
  tex_params.sampleable = true;
  tex_params.renderable = true;
  tex_params.host_readable = true;
  tex_params.blit_dst = detail::has_all(format->caps, detail::flag_bits(PL_FMT_CAP_BLITTABLE));
  tex_params.debug_tag = "ajp hors écran";
  impl->tex = pl_tex_create(gpu, &tex_params);
  if (impl->tex == nullptr)
    return std::unexpected(Refusal{RefusalCode::RenderFailed, "pl_tex_create (hors écran)"});

  const std::size_t bytes = impl->values() * sizeof(std::uint16_t);
  auto buf_params = detail::zeroed<pl_buf_params>();
  buf_params.size = bytes;
  buf_params.host_mapped = bytes <= gpu->limits.max_mapped_size;
  buf_params.host_readable = !buf_params.host_mapped;
  buf_params.debug_tag = "ajp lecture hors écran";
  impl->buf = pl_buf_create(gpu, &buf_params);
  if (impl->buf == nullptr)
    return std::unexpected(Refusal{RefusalCode::RenderFailed, "pl_buf_create (hors écran)"});
  if (!buf_params.host_mapped) impl->host_copy.resize(impl->values());

  pl_frame& frame = impl->frame;
  frame.num_planes = 1;
  frame.planes[0].texture = impl->tex;
  frame.planes[0].components = 4;
  for (int c = 0; c < 4; ++c) frame.planes[0].component_mapping[c] = c;
  frame.repr = pl_color_repr_rgb;
  frame.repr.bits = {.sample_depth = 16, .color_depth = 16};
  frame.color = detail::to_placebo(options.display);
  frame.crop = {.x0 = 0.0F,
                .y0 = 0.0F,
                .x1 = static_cast<float>(options.width),
                .y1 = static_cast<float>(options.height)};
  if (tex_params.blit_dst) pl_tex_clear(gpu, impl->tex, std::array<float, 4>{0.0F, 0.0F, 0.0F, 1.0F}.data());
  return OffscreenTarget(std::move(impl));
}

OffscreenTarget::OffscreenTarget(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
OffscreenTarget::OffscreenTarget(OffscreenTarget&&) noexcept = default;
OffscreenTarget& OffscreenTarget::operator=(OffscreenTarget&&) noexcept = default;
OffscreenTarget::~OffscreenTarget() = default;

const pl_frame& OffscreenTarget::frame() const {
  return impl_->frame;
}

const pl_tex_t* OffscreenTarget::texture() const {
  return impl_->tex;
}

int OffscreenTarget::width() const {
  return impl_->tex->params.w;
}

int OffscreenTarget::height() const {
  return impl_->tex->params.h;
}

std::expected<void, Refusal> OffscreenTarget::start_readback() {
  Impl& s = *impl_;
  if (s.state == Impl::Readback::Pending && pl_buf_poll(s.gpu, s.buf, 0)) {
    return std::unexpected(Refusal{RefusalCode::SequenceOrder, "copie hors écran encore en cours"});
  }
  pl_tex_transfer_params transfer{};
  transfer.tex = s.tex;
  transfer.buf = s.buf;
  transfer.buf_offset = 0;
  transfer.row_pitch = static_cast<std::size_t>(s.tex->params.w) * kChannels * sizeof(std::uint16_t);
  if (!pl_tex_download(s.gpu, &transfer)) {
    s.state = Impl::Readback::None;
    return std::unexpected(Refusal{RefusalCode::RenderFailed, "pl_tex_download (hors écran)"});
  }
  pl_gpu_flush(s.gpu);
  s.state = Impl::Readback::Pending;
  return {};
}

bool OffscreenTarget::readback_ready(std::uint64_t timeout_ns) {
  Impl& s = *impl_;
  if (s.state == Impl::Readback::Pending && !pl_buf_poll(s.gpu, s.buf, timeout_ns))
    s.state = Impl::Readback::Ready;
  return s.state == Impl::Readback::Ready;
}

std::expected<std::span<const std::uint16_t>, Refusal> OffscreenTarget::pixels() {
  Impl& s = *impl_;
  if (!readback_ready())
    return std::unexpected(Refusal{RefusalCode::SequenceOrder, "aucune copie hors écran arrivée"});
  if (s.buf->data != nullptr) {
    // Le mapping persistant est aligné pour tout type : on le lit en uint16 sans copie.
    return std::span<const std::uint16_t>(reinterpret_cast<const std::uint16_t*>(s.buf->data), s.values());
  }
  if (!pl_buf_read(s.gpu, s.buf, 0, s.host_copy.data(), s.host_copy.size() * sizeof(std::uint16_t))) {
    return std::unexpected(Refusal{RefusalCode::RenderFailed, "pl_buf_read (hors écran)"});
  }
  return std::span<const std::uint16_t>(s.host_copy);
}

}  // namespace ajp::render
