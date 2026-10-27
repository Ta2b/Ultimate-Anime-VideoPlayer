#include "app/play/placebo_screen.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <format>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <SDL3/SDL_filesystem.h>
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

// Blanc de l'interface dans une sortie HDR : le blanc de référence (BT.2408), comme le SDR.
constexpr float kUiWhiteNits = 203.0F;

// Images gardées pour le profil GPU du grade (une heure à 165 Hz tient en dessous).
constexpr std::size_t kMaxGradeSamples = 1 << 20;

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

}  // namespace

PlaceboScreen::PlaceboScreen(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

PlaceboScreen::~PlaceboScreen() = default;

const GpuDevice& PlaceboScreen::device() const {
  return impl_->device;
}

const media::HwDeviceSource* PlaceboScreen::shared_devices() const {
  return impl_->share.get();
}

AVBufferRef* PlaceboScreen::vulkan_device() const {
  return impl_->share ? impl_->share->vulkan() : nullptr;
}

SDL_Window* PlaceboScreen::window() const {
  return impl_->window;
}

std::string PlaceboScreen::output_description() const {
  return impl_->hdr_output ? "sortie HDR10 (PQ BT.2020)" : "sortie SDR";
}

std::expected<std::vector<RingSlotDesc>, Refusal> PlaceboScreen::create_ring(int width, int height,
                                                                             std::uint32_t count,
                                                                             bool external) {
  return impl_->ring.create(impl_->vk->gpu, width, height, count, external);
}

bool PlaceboScreen::hdr_output() const {
  return impl_->hdr_output;
}

void PlaceboScreen::flush() {
  pl_gpu_flush(impl_->vk->gpu);
}

void PlaceboScreen::snapshot_next(std::filesystem::path path, int count) {
  impl_->snapshot = std::move(path);
  impl_->snapshot_count = std::max(count, 1);
  impl_->snapshot_index = 0;
}

void PlaceboScreen::release_ui(ImGuiPlatformIO& platform) {
  if (impl_->ui) impl_->ui->release_textures(platform);
}

void PlaceboScreen::Impl::draw_ui(const ImDrawData* data, const pl_frame& target) {
  if (data == nullptr || ui_failed) return;
  if (auto drawn = ui->draw(*data, target, kUiWhiteNits); !drawn) {
    // Un échec isolé (texture ou tampon d'un redimensionnement) ne retire pas l'interface : elle
    // n'est abandonnée qu'après des échecs de suite.
    ++ui_failures;
    ui_failed = ui_failures >= kMaxUiFailures;
    if (ui_failures == 1 || ui_failed)
      channel->warn("interface : {} {}{}", name(drawn.error().code), drawn.error().detail,
                    ui_failed ? " (abandonnée)" : "");
  } else {
    ui_failures = 0;
  }
}

std::expected<void, Refusal> PlaceboScreen::Impl::compose(const pl_frame* image,
                                                          const pl_frame& target) const {
  pl_render_params params = pl_render_default_params;
  // Le grade chargé est celui du PQ ou du SDR : sur une autre cible (instantané SDR d'une sortie
  // HDR10), il est sauté.
  const bool graded_now = grade.has_value() && graded(target.color);
  if (graded_now) {
    const std::span<const pl_hook* const> hooks = grade->hooks();
    params.hooks = hooks.data();
    params.num_hooks = static_cast<int>(hooks.size());
    params.info_callback = [](void* priv, const pl_render_info* info) {
      const char* what = info->pass->shader->description;
      if (what != nullptr && std::string_view(what).find("AnimeJaNai") != std::string_view::npos)
        static_cast<const Impl*>(priv)->grade_frame_ns += info->pass->last;
    };
    params.info_priv = const_cast<Impl*>(this);
    grade_frame_ns = 0;
  }
  if (image == nullptr) pl_frame_clear(vk->gpu, &target, std::array<float, 3>{0.0F, 0.0F, 0.0F}.data());
  if (!pl_render_image(renderer, image, &target, &params)) {
    return std::unexpected(Refusal{RefusalCode::VendorCallFailed, "pl_render_image a échoué"});
  }
  if (graded_now && image != nullptr && grade_frame_ns > 0 && grade_gpu_us.size() < kMaxGradeSamples)
    grade_gpu_us.push_back(static_cast<std::uint32_t>(grade_frame_ns / 1000));
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

bool PlaceboScreen::ready(const RingFrame& frame) const {
  return impl_->ring.ready(frame);
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
  s.target_color = swframe.color_space;
  s.select_grade(s.target_color);
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
    pl_rect2df_aspect_set(&target.crop, pl_rect2df_aspect(&image.crop) * static_cast<float>(s.sample_aspect),
                          0.0F);
  }

  auto overlays = s.overlays->build(request.subtitles, request.osd.empty() ? nullptr : &request.osd, *s.osd,
                                    out, request.ui != nullptr ? request.ui_bottom : 0,
                                    request.ui != nullptr ? request.ui_top : 0);
  if (!overlays) return std::unexpected(std::move(overlays.error()));
  target.overlays = overlays->data();
  target.num_overlays = static_cast<int>(overlays->size());

  if (auto drawn = s.compose(request.frame != nullptr ? &image : nullptr, target); !drawn) {
    return std::unexpected(std::move(drawn.error()));
  }
  s.draw_ui(request.ui, target);
  if (s.snapshot) {
    std::filesystem::path path = *s.snapshot;
    if (s.snapshot_count > 1) {
      // « nom.pq.ppm » → « nom-001.pq.ppm », « nom.ppm » → « nom-001.ppm ».
      const std::string file = path.filename().string();
      const std::size_t dot = file.ends_with(".pq.ppm") ? file.size() - 7 : file.rfind('.');
      path.replace_filename(std::format("{}-{:03}{}", file.substr(0, dot), s.snapshot_index + 1,
                                        dot == std::string::npos ? std::string() : file.substr(dot)));
    }
    if (++s.snapshot_index >= s.snapshot_count) s.snapshot.reset();
    if (auto saved =
            s.save_snapshot(request.frame != nullptr ? &image : nullptr, target, out, request.ui, path);
        !saved) {
      s.channel->warn("instantané {} : {}", path.string(), saved.error().detail);
    }
  }
  if (!pl_swapchain_submit_frame(s.swapchain))
    return refuse(RefusalCode::VendorCallFailed, "pl_swapchain_submit_frame");
  const std::int64_t submitted = host_now_us();
  pl_swapchain_swap_buffers(s.swapchain);  // FIFO : attend une place en file quand elle est pleine
  const std::int64_t presented = host_now_us();
  return ScreenTiming{.render_us = submitted - start,
                      .present_us = presented,
                      .queue_wait_us = presented - submitted,
                      .presented = true};
}

}  // namespace ajp::app::play
