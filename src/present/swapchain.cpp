#include "present/swapchain.h"

#include <format>
#include <utility>

#include <libplacebo/gpu.h>
#include <libplacebo/renderer.h>
#include <libplacebo/swapchain.h>
#include <libplacebo/vulkan.h>

#include "core/clock.h"
#include "present/frame_vsyncs.h"
#include "present/log.h"
#include "present/vk_handle.h"

namespace ajp::present {

namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

SurfaceTransfer transfer_of(const pl_color_space& space) {
  if (space.transfer == PL_COLOR_TRC_PQ) return SurfaceTransfer::Pq;
  if (space.transfer == PL_COLOR_TRC_LINEAR) return SurfaceTransfer::Linear;
  return SurfaceTransfer::Other;
}

std::string_view space_name(core::OutputSpace space) {
  switch (space) {
    case core::OutputSpace::Srgb:
      return "SDR sRGB";
    case core::OutputSpace::Pq:
      return "HDR10 PQ BT.2020";
    case core::OutputSpace::ScRgb:
      return "scRGB linéaire";
  }
  return "inconnu";
}

}  // namespace

struct Swapchain::Impl {
  pl_swapchain swapchain = nullptr;
  pl_gpu gpu = nullptr;          // celui du pl_vulkan de create(), pour reconnaître sa perte
  SDL_Window* window = nullptr;  // celle de create(), qui survit à la swapchain
  pl_frame target{};
  VsyncEstimator vsync;
  OutputState output;
  OutputHint hint = OutputHint::Sdr;
  float peak_nits = 0.0F;
  float min_nits = 0.0F;
  bool fallback_tried = false;
  FrameVsyncs frames;
  std::shared_ptr<spdlog::logger> log = make_logger();

  ~Impl() { pl_swapchain_destroy(&swapchain); }
  Impl() = default;
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  void apply_hint(OutputHint next) {
    hint = next;
    switch (next) {
      case OutputHint::Sdr:
        pl_swapchain_colorspace_hint(swapchain, nullptr);
        return;
      case OutputHint::Pq: {
        const pl_color_space pq{
            .primaries = PL_COLOR_PRIM_BT_2020,
            .transfer = PL_COLOR_TRC_PQ,
            .hdr = {.min_luma = min_nits, .max_luma = peak_nits, .max_cll = peak_nits},
        };
        pl_swapchain_colorspace_hint(swapchain, &pq);
        return;
      }
      case OutputHint::ScRgb: {
        const pl_color_space linear{
            .primaries = PL_COLOR_PRIM_BT_709,
            .transfer = PL_COLOR_TRC_LINEAR,
            .hdr = {.min_luma = min_nits, .max_luma = peak_nits, .max_cll = peak_nits},
        };
        pl_swapchain_colorspace_hint(swapchain, &linear);
        return;
      }
    }
  }
};

std::expected<Swapchain, Refusal> Swapchain::create(const Window& window, const Surface& surface,
                                                    const pl_vulkan_t* vk, const SwapchainOptions& options) {
  if (!swapchain_limits::kDepth.contains(options.depth)) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("profondeur de swapchain {} hors de [1, 8]", options.depth));
  }
  if (vk == nullptr || surface.handle() == 0) {
    return refuse(RefusalCode::PresentFailed, "pl_vulkan_create_swapchain : appareil ou surface absents");
  }
  auto impl = std::make_unique<Impl>();
  // Pas la macro pl_vulkan_swapchain_params : un littéral composé, interdit en C++.
  const pl_vulkan_swapchain_params params{
      .surface = from_u64<VkSurfaceKHR>(surface.handle()),
      .present_mode = VK_PRESENT_MODE_FIFO_KHR,
      .swapchain_depth = options.depth,
  };
  impl->window = window.sdl();
  impl->swapchain = pl_vulkan_create_swapchain(vk, &params);
  if (impl->swapchain == nullptr) {
    return refuse(RefusalCode::PresentFailed,
                  "pl_vulkan_create_swapchain : refusée (VK_KHR_swapchain absente ou surface inutilisable, "
                  "voir le journal de libplacebo)");
  }
  impl->gpu = vk->gpu;
  Swapchain swapchain(std::move(impl));
  const auto size = window.pixel_size();
  if (!size) return std::unexpected(size.error());
  if (auto sized = swapchain.resize(*size); !sized) return std::unexpected(sized.error());
  return swapchain;
}

Swapchain::Swapchain(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Swapchain::Swapchain(Swapchain&& other) noexcept = default;
Swapchain& Swapchain::operator=(Swapchain&& other) noexcept = default;
Swapchain::~Swapchain() = default;

std::expected<void, Refusal> Swapchain::request_output(const core::DisplayInfo& display,
                                                       double grade_peak_nits) {
  if (!core::limits::kPeakNits.contains(grade_peak_nits)) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("pic du grade {} nits hors de [1, 10000]", grade_peak_nits));
  }
  if (auto valid = core::check_display(display); !valid) return std::unexpected(valid.error());
  impl_->peak_nits = static_cast<float>(grade_peak_nits);
  impl_->min_nits = static_cast<float>(display.min_nits);
  impl_->fallback_tried = false;
  impl_->apply_hint(first_hint(display));
  return {};
}

std::expected<PixelSize, Refusal> Swapchain::resize(PixelSize size) {
  if (!window_limits::kSide.contains(size.width) || !window_limits::kSide.contains(size.height)) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("swapchain {}x{} hors de [1, 16384]", size.width, size.height));
  }
  PixelSize kept = size;
  if (!pl_swapchain_resize(impl_->swapchain, &kept.width, &kept.height)) {
    return refuse(RefusalCode::PresentFailed,
                  std::format("pl_swapchain_resize({}x{}) : swapchain non recréée", size.width, size.height));
  }
  return kept;
}

std::expected<std::optional<SwapFrame>, Refusal> Swapchain::start_frame() {
  Impl& p = *impl_;
  if (!Window::presentable(p.window)) return std::nullopt;
  pl_swapchain_frame frame{};
  if (!pl_swapchain_start_frame(p.swapchain, &frame)) {
    // libplacebo ne dit pas pourquoi : un appareil perdu ne reviendra pas, le reste (swapchain
    // recréée, surface sans taille) se réessaie.
    if (pl_gpu_is_failed(p.gpu)) {
      return refuse(RefusalCode::PresentFailed, "pl_swapchain_start_frame : appareil Vulkan perdu");
    }
    return std::nullopt;
  }
  const std::int64_t acquired = host_now_us();
  p.vsync.add(acquired);

  const SurfaceTransfer transfer = transfer_of(frame.color_space);
  const OutputState state = classify_output(p.hint, transfer);
  if (state != p.output) {
    p.log->info("sortie : {}{}{}", space_name(state.space), state.grade ? "" : ", grade coupé",
                state.hdr_lost ? " (HDR10 refusé par la surface)" : "");
    p.output = state;
  }
  if (!p.fallback_tried) {
    if (const auto next = fallback_hint(p.hint, transfer)) {
      p.log->warn("HDR10 PQ refusé par la surface : essai du repli scRGB");
      p.fallback_tried = true;
      p.apply_hint(*next);
    }
  }
  pl_frame_from_swapchain(&p.target, &frame);
  const VsyncTick tick = p.frames.next(p.vsync, acquired, pl_swapchain_latency(p.swapchain));
  return SwapFrame{
      .target = &p.target,
      .size = {.width = frame.fbo->params.w, .height = frame.fbo->params.h},
      .flipped = frame.flipped,
      .output = state,
      .acquired_host_us = acquired,
      .tick = tick,
  };
}

std::expected<void, Refusal> Swapchain::submit_frame() {
  if (!pl_swapchain_submit_frame(impl_->swapchain)) {
    return refuse(RefusalCode::PresentFailed, "pl_swapchain_submit_frame : appareil ou surface perdus");
  }
  pl_swapchain_swap_buffers(impl_->swapchain);
  return {};
}

const VsyncEstimator& Swapchain::vsync() const {
  return impl_->vsync;
}

OutputState Swapchain::output() const {
  return impl_->output;
}

int Swapchain::latency() const {
  return pl_swapchain_latency(impl_->swapchain);
}

}  // namespace ajp::present
