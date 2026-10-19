#include "app/play/placebo_screen.h"

#include <format>
#include <utility>

#include <SDL3/SDL_video.h>
#include <SDL3/SDL_vulkan.h>
#include <libplacebo/log.h>
#include <libplacebo/renderer.h>
#include <libplacebo/swapchain.h>
#include <libplacebo/vulkan.h>

#include "app/play/osd_text.h"
#include "app/play/placebo_impl.h"
#include "core/clock.h"
#include "core/log.h"

namespace ajp::app::play {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

void placebo_log(void* priv, pl_log_level level, const char* message) {
  auto* log = static_cast<spdlog::logger*>(priv);
  if (level <= PL_LOG_ERR) {
    log->error("libplacebo : {}", message);
  } else if (level == PL_LOG_WARN) {
    log->warn("libplacebo : {}", message);
  }
}

pl_color_space color_space_of(const ColorInfo& color) {
  pl_color_space space{};
  switch (color.primaries) {
    case Primaries::Bt709:
      space.primaries = PL_COLOR_PRIM_BT_709;
      break;
    case Primaries::Bt601_625:
      space.primaries = PL_COLOR_PRIM_BT_601_625;
      break;
    case Primaries::Bt601_525:
      space.primaries = PL_COLOR_PRIM_BT_601_525;
      break;
    case Primaries::Bt2020:
      space.primaries = PL_COLOR_PRIM_BT_2020;
      break;
    case Primaries::DciP3:
      space.primaries = PL_COLOR_PRIM_DCI_P3;
      break;
  }
  switch (color.transfer) {
    case Transfer::Bt1886:
      space.transfer = PL_COLOR_TRC_BT_1886;
      break;
    case Transfer::Srgb:
      space.transfer = PL_COLOR_TRC_SRGB;
      break;
    case Transfer::Linear:
      space.transfer = PL_COLOR_TRC_LINEAR;
      break;
    case Transfer::Pq:
      space.transfer = PL_COLOR_TRC_PQ;
      break;
    case Transfer::Hlg:
      space.transfer = PL_COLOR_TRC_HLG;
      break;
  }
  return space;
}

}  // namespace

PlaceboScreen::PlaceboScreen(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

PlaceboScreen::~PlaceboScreen() = default;

PlaceboScreen::Impl::~Impl() {
  if (vk != nullptr) pl_gpu_finish(vk->gpu);
  osd.reset();
  overlays.reset();
  ring.clear(vk != nullptr ? vk->gpu : nullptr);
  pl_renderer_destroy(&renderer);
  pl_swapchain_destroy(&swapchain);
  pl_vulkan_destroy(&vk);
  if (surface != VK_NULL_HANDLE) SDL_Vulkan_DestroySurface(inst->instance, surface, nullptr);
  pl_vk_inst_destroy(&inst);
  pl_log_destroy(&log);
  if (window != nullptr) SDL_DestroyWindow(window);
}

std::expected<std::unique_ptr<PlaceboScreen>, Refusal> PlaceboScreen::open(const ScreenOptions& options) {
  auto impl = std::make_unique<Impl>();
  impl->channel = log_channel("app");
  SDL_WindowFlags flags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY;
  if (options.fullscreen) flags |= SDL_WINDOW_FULLSCREEN;
  impl->window = SDL_CreateWindow(options.title.c_str(), options.width, options.height, flags);
  if (impl->window == nullptr) {
    return refuse(RefusalCode::LibraryUnavailable, std::format("SDL_CreateWindow : {}", SDL_GetError()));
  }
  impl->fullscreen = options.fullscreen;

  pl_log_params log_params{};
  log_params.log_cb = placebo_log;
  log_params.log_priv = impl->channel.get();
  log_params.log_level = PL_LOG_WARN;
  impl->log = pl_log_create(PL_API_VER, &log_params);

  Uint32 count = 0;
  const char* const* sdl_extensions = SDL_Vulkan_GetInstanceExtensions(&count);
  if (sdl_extensions == nullptr) {
    return refuse(RefusalCode::LibraryUnavailable,
                  std::format("SDL_Vulkan_GetInstanceExtensions : {}", SDL_GetError()));
  }
  pl_vk_inst_params inst_params{};
  inst_params.extensions = sdl_extensions;
  inst_params.num_extensions = static_cast<int>(count);
  impl->inst = pl_vk_inst_create(impl->log, &inst_params);
  if (impl->inst == nullptr) return refuse(RefusalCode::LibraryUnavailable, "pl_vk_inst_create a échoué");
  if (!SDL_Vulkan_CreateSurface(impl->window, impl->inst->instance, nullptr, &impl->surface)) {
    return refuse(RefusalCode::LibraryUnavailable,
                  std::format("SDL_Vulkan_CreateSurface : {}", SDL_GetError()));
  }

  pl_vulkan_params vk_params = pl_vulkan_default_params;
  vk_params.instance = impl->inst->instance;
  vk_params.get_proc_addr = impl->inst->get_proc_addr;
  vk_params.surface = impl->surface;
  vk_params.allow_software = false;
  impl->vk = pl_vulkan_create(impl->log, &vk_params);
  if (impl->vk == nullptr)
    return refuse(RefusalCode::DeviceMismatch, "pl_vulkan_create : aucun appareil ne convient");
  const pl_gpu gpu = impl->vk->gpu;
  if ((gpu->export_caps.tex & PL_HANDLE_FD) == 0 || (gpu->export_caps.sync & PL_HANDLE_FD) == 0) {
    return refuse(RefusalCode::DeviceMismatch, "l'appareil n'exporte pas ses images ni ses sémaphores (fd)");
  }

  pl_vulkan_swapchain_params sw_params{};
  sw_params.surface = impl->surface;
  sw_params.present_mode = VK_PRESENT_MODE_FIFO_KHR;
  sw_params.swapchain_depth = 3;
  impl->swapchain = pl_vulkan_create_swapchain(impl->vk, &sw_params);
  if (impl->swapchain == nullptr)
    return refuse(RefusalCode::LibraryUnavailable, "pl_vulkan_create_swapchain a échoué");
  if (options.hdr) pl_swapchain_colorspace_hint(impl->swapchain, &pl_color_space_hdr10);
  impl->renderer = pl_renderer_create(impl->log, gpu);

  impl->device = GpuDevice{.api = GpuApi::Vulkan, .name = "Vulkan (libplacebo)"};
  for (std::size_t i = 0; i < impl->device.uuid.size(); ++i) impl->device.uuid[i] = gpu->uuid[i];
  auto osd = OsdText::create();
  if (!osd) return std::unexpected(std::move(osd.error()));
  impl->osd = std::move(*osd);
  impl->overlays = std::make_unique<Overlays>(gpu);
  impl->source = pl_color_space_bt709;
  return std::unique_ptr<PlaceboScreen>(new PlaceboScreen(std::move(impl)));
}

const GpuDevice& PlaceboScreen::device() const {
  return impl_->device;
}
SDL_Window* PlaceboScreen::window() const {
  return impl_->window;
}

std::string PlaceboScreen::output_description() const {
  return impl_->hdr_output ? "sortie HDR10 (PQ BT.2020)" : "sortie SDR";
}

std::expected<std::vector<RingSlotDesc>, Refusal> PlaceboScreen::create_ring(int width, int height,
                                                                             std::uint32_t count) {
  return impl_->ring.create(impl_->vk->gpu, width, height, count);
}

void PlaceboScreen::set_source_color(const ColorInfo& color) {
  impl_->source = color_space_of(color);
}

void PlaceboScreen::flush() {
  pl_gpu_flush(impl_->vk->gpu);
}

void PlaceboScreen::snapshot_next(std::filesystem::path path) {
  impl_->snapshot = std::move(path);
}

std::expected<void, Refusal> PlaceboScreen::Impl::compose(const pl_frame* image, const pl_frame& target) {
  const pl_render_params params = pl_render_default_params;
  if (image == nullptr) pl_frame_clear(vk->gpu, &target, std::array<float, 3>{0.0F, 0.0F, 0.0F}.data());
  if (!pl_render_image(renderer, image, &target, &params)) {
    return std::unexpected(Refusal{RefusalCode::VendorCallFailed, "pl_render_image a échoué"});
  }
  return {};
}

OutputSize PlaceboScreen::size() const {
  int w = 0;
  int h = 0;
  SDL_GetWindowSizeInPixels(impl_->window, &w, &h);
  return {w, h};
}

void PlaceboScreen::set_fullscreen(bool on) {
  if (SDL_SetWindowFullscreen(impl_->window, on)) impl_->fullscreen = on;
}

bool PlaceboScreen::fullscreen() const {
  return impl_->fullscreen;
}

SlotFree PlaceboScreen::retire(const RingFrame& frame) {
  return impl_->ring.retire(impl_->vk->gpu, frame);
}

std::expected<ScreenTiming, Refusal> PlaceboScreen::show(const ShowRequest& request) {
  Impl& s = *impl_;
  const pl_gpu gpu = s.vk->gpu;
  const std::int64_t start = host_now_us();
  OutputSize out = size();
  if (out != s.swapchain_size) {
    int w = out.width;
    int h = out.height;
    if (!pl_swapchain_resize(s.swapchain, &w, &h)) return ScreenTiming{.presented = false};
    s.swapchain_size = out;
  }
  pl_swapchain_frame swframe{};
  if (!pl_swapchain_start_frame(s.swapchain, &swframe))
    return ScreenTiming{.presented = false};  // fenêtre cachée
  s.hdr_output = swframe.color_space.transfer == PL_COLOR_TRC_PQ;
  pl_frame target{};
  pl_frame_from_swapchain(&target, &swframe);

  pl_frame image{};
  if (request.frame != nullptr) {
    auto tex = s.ring.acquire(gpu, *request.frame);
    if (!tex) return std::unexpected(std::move(tex.error()));
    image.num_planes = 1;
    image.planes[0].texture = *tex;
    image.planes[0].components = 3;
    image.planes[0].component_mapping[0] = 0;
    image.planes[0].component_mapping[1] = 1;
    image.planes[0].component_mapping[2] = 2;
    image.repr = pl_color_repr_rgb;
    image.color = s.source;
    image.crop = {0, 0, static_cast<float>((*tex)->params.w), static_cast<float>((*tex)->params.h)};
    pl_rect2df_aspect_copy(&target.crop, &image.crop, 0.0F);
  }

  auto overlays =
      s.overlays->build(request.subtitles, request.osd.empty() ? nullptr : &request.osd, *s.osd, out);
  if (!overlays) return std::unexpected(std::move(overlays.error()));
  target.overlays = overlays->data();
  target.num_overlays = static_cast<int>(overlays->size());

  if (auto drawn = s.compose(request.frame != nullptr ? &image : nullptr, target); !drawn) {
    return std::unexpected(std::move(drawn.error()));
  }
  if (s.snapshot) {
    if (auto saved = s.save_snapshot(request.frame != nullptr ? &image : nullptr, target, out); !saved) {
      s.channel->warn("instantané {} : {}", s.snapshot->string(), saved.error().detail);
    }
    s.snapshot.reset();
  }
  if (!pl_swapchain_submit_frame(s.swapchain))
    return refuse(RefusalCode::VendorCallFailed, "pl_swapchain_submit_frame");
  const std::int64_t submitted = host_now_us();
  pl_swapchain_swap_buffers(s.swapchain);  // FIFO : rythme de l'écran
  return ScreenTiming{.render_us = submitted - start, .present_us = host_now_us(), .presented = true};
}

}  // namespace ajp::app::play
