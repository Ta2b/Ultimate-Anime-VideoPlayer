#include "app/play/placebo_screen.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
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

#include "app/play/calibration_shot.h"
#include "app/play/osd_text.h"
#include "app/play/placebo_impl.h"
#include "core/clock.h"
#include "core/log.h"
#include "core/trace.h"

namespace ajp::app::play {
namespace {
// Observations de la grille des vblanks avant de s'y fier pour la synchro (≈ 0,7 s à 165 Hz).
constexpr std::size_t kGridSamples = 120;
}  // namespace
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
                                                                             bool external,
                                                                             RingFormat format) {
  return impl_->ring.create(impl_->vk->gpu, width, height, count, external, format);
}

bool PlaceboScreen::hdr_output() const {
  return impl_->hdr_output;
}

void PlaceboScreen::new_scene() {
  if (impl_->renderer != nullptr) pl_renderer_flush_cache(impl_->renderer);
  ++impl_->grade_serial;
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
  AJP_TRACE("composition");
  pl_render_params params = pl_render_default_params;
  switch (final_scaler) {
    case FinalScaler::Default:
      break;
    case FinalScaler::EwaLanczos:
      params.upscaler = &pl_filter_ewa_lanczos;
      break;
    case FinalScaler::EwaLanczosSharp:
      params.upscaler = &pl_filter_ewa_lanczossharp;
      break;
  }
  // Le grade chargé est celui du PQ ou du SDR : sur une autre cible (instantané SDR d'une sortie
  // HDR10), il est sauté.
  const bool graded_now = !ring_image && grade.has_value() && graded(target.color);
  // Le grade étire la source (plus de nits, plus de couleur) : les paliers d'un dégradé 8 bits, invisibles en
  // SDR, ressortent en bandes dans un halo ou un ciel, mais le débandage de libplacebo (échantillonnage
  // aléatoire) fabriquait du bruit dans les aplats sombres que le grade étire (mesuré : écart-type 16x16 de
  // 1,0 contre 0,0). Il n'est donc plus posé d'office (--grade-deband le rétablit) ; le débandage demandé à
  // la main reste, sans grain sous le grade, et une source déjà débandée avant SPAN n'est pas retouchée.
  static const pl_deband_params kGradeDeband = [] {
    pl_deband_params p = pl_deband_default_params;
    p.grain = 0.0F;
    return p;
  }();
  // Le grain de libplacebo (4 codes) serait étiré par le grade, surtout dans les ombres et sur les peaux :
  // sous le grade, même le débandage demandé à la main part sans grain.
  // Mesuré : même sans grain, ce débandage fabrique du bruit dans les aplats sombres que le grade étire ;
  // sous le grade il n'est donc posé (réglage manuel compris) que par --grade-deband.
  if (ring_image) {
    // Gradée dans l'anneau (chantier 2c) : ni hooks, ni détection de pic (l'image déclare la cible de sa
    // version), et le débandage seulement s'il est demandé, sans grain.
    params.peak_detect_params = nullptr;
    if (deband) params.deband_params = &kGradeDeband;
  } else if (graded_now) {
    if (grade_deband && (deband || !source_debanded)) params.deband_params = &kGradeDeband;
  } else if (deband) {
    params.deband_params = &pl_deband_default_params;
  }
  if (graded_now) {
    const std::span<const pl_hook* const> hooks = grade->hooks();
    params.hooks = hooks.data();
    params.num_hooks = static_cast<int>(hooks.size());
  }
  // Le temps GPU de chaque passe, toujours relevé (libplacebo le rend avec un retard d'une image ou deux) :
  // toutes les passes pour l'affichage, celles du grade à part.
  params.info_callback = [](void* priv, const pl_render_info* info) {
    Impl& self = *static_cast<Impl*>(priv);
    self.display_frame_ns += info->pass->last;
    const char* what = info->pass->shader->description;
    if (what != nullptr && std::string_view(what).find("AnimeJaNai") != std::string_view::npos)
      self.grade_frame_ns += info->pass->last;
  };
  params.info_priv = const_cast<Impl*>(this);
  grade_frame_ns = 0;
  display_frame_ns = 0;
  if (image == nullptr) pl_frame_clear(vk->gpu, &target, std::array<float, 3>{0.0F, 0.0F, 0.0F}.data());
  if (!pl_render_image(renderer, image, &target, &params)) {
    return std::unexpected(Refusal{RefusalCode::VendorCallFailed, "pl_render_image a échoué"});
  }
  if (graded_now && image != nullptr && grade_frame_ns > 0 && grade_gpu_us.size() < kMaxGradeSamples)
    grade_gpu_us.push_back(static_cast<std::uint32_t>(grade_frame_ns / 1000));
  return {};
}

OutputSize PlaceboScreen::size() const {
  if (impl_->offscreen) return impl_->offscreen_size;
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

void PlaceboScreen::watch_writes(void (*waker)(void*), void* context) {
  Impl& s = *impl_;
  s.ring.set_watcher(nullptr);
  s.watcher.reset();
  if (waker == nullptr || s.vk == nullptr) return;
  const auto device_proc =
      reinterpret_cast<PFN_vkGetDeviceProcAddr>(s.vk->get_proc_addr(s.vk->instance, "vkGetDeviceProcAddr"));
  s.watcher = RingWatcher::start(s.vk->device, device_proc, waker, context);
  if (!s.watcher) s.channel->warn("veilleur d'écriture indisponible : main sonde l'anneau comme avant");
  s.ring.set_watcher(s.watcher.get());
}

void PlaceboScreen::watch(const RingFrame& frame) {
  if (impl_->watcher) impl_->watcher->watch(impl_->ring.semaphore_of(frame), frame.ready_value);
}

std::optional<std::int64_t> PlaceboScreen::written_at(const RingFrame& frame) {
  if (!impl_->watcher) return std::nullopt;
  return impl_->watcher->take(impl_->ring.semaphore_of(frame), frame.ready_value);
}

bool PlaceboScreen::wakes_on_write() const {
  return impl_->watcher != nullptr;
}

std::optional<bool> PlaceboScreen::vrr_armed() const {
  if (!impl_->feedback) return std::nullopt;
  switch (impl_->feedback->refresh_state()) {
    case RefreshState::Variable:
      return true;
    case RefreshState::Fixed:
      return impl_->present_level == PresentLevel::Wayland ? std::nullopt : std::optional<bool>(false);
    case RefreshState::Unknown:
      break;
  }
  return std::nullopt;
}

bool PlaceboScreen::vrr_changed() {
  return impl_->feedback && impl_->feedback->take_refresh_change();
}

void PlaceboScreen::set_contract(const render::DisplayContract& contract) {
  impl_->contract = contract;
}

const std::optional<render::DisplayContract>& PlaceboScreen::contract() const {
  return impl_->contract;
}

std::optional<core::VblankGrid> PlaceboScreen::vblank_grid() const {
  const PresentFeedback* feedback = impl_->feedback.get();
  if (feedback == nullptr || !feedback->dated() || !feedback->fixed_refresh()) return std::nullopt;
  if (feedback->clock().samples() < kGridSamples) return std::nullopt;
  return feedback->clock().grid();
}

std::expected<bool, Refusal> PlaceboScreen::Impl::compose_kept(const pl_frame* image, const pl_frame& target,
                                                               const ComposeKey& key,
                                                               std::span<const pl_overlay> subtitles) {
  const pl_gpu gpu = vk->gpu;
  if (target.num_planes != 1 || target.planes[0].texture == nullptr) return false;
  const pl_tex out = target.planes[0].texture;
  const pl_fmt format = out->params.format;
  constexpr auto kNeeded = static_cast<pl_fmt_caps>(PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_BLITTABLE);
  if (!out->params.blit_dst || (format->caps & kNeeded) != kNeeded) {
    composed_key.reset();
    return false;
  }
  const bool same_tex = composed != nullptr && composed->params.w == out->params.w &&
                        composed->params.h == out->params.h && composed->params.format == format;
  if (!same_tex || !reusable(composed_key, key)) {
    composed_key.reset();
    pl_tex_params params{};
    params.w = out->params.w;
    params.h = out->params.h;
    params.format = format;
    params.renderable = true;
    params.blit_src = true;
    if (!pl_tex_recreate(gpu, &composed, &params)) return false;  // la composition directe reste
    pl_frame into = target;
    into.planes[0].texture = composed;
    into.overlays = subtitles.data();
    into.num_overlays = static_cast<int>(subtitles.size());
    if (auto drawn = compose(image, into); !drawn) return std::unexpected(std::move(drawn.error()));
    composed_key = key;
    ++recompositions;
  } else {
    // Rien de recomposé : les passes de cette présentation ne sont que la copie et l'OSD.
    display_frame_ns = 0;
    ++reuses;
  }
  pl_tex_blit_params blit{};
  blit.src = composed;
  blit.dst = out;
  pl_tex_blit(gpu, &blit);
  return true;
}

std::expected<void, Refusal> PlaceboScreen::Impl::draw_overlays(const pl_frame& target,
                                                                std::span<const pl_overlay> layers) const {
  if (layers.empty()) return {};
  pl_frame on = target;
  on.overlays = layers.data();
  on.num_overlays = static_cast<int>(layers.size());
  if (!pl_render_image(renderer, nullptr, &on, &pl_render_default_params))
    return std::unexpected(Refusal{RefusalCode::VendorCallFailed, "pl_render_image (OSD) a échoué"});
  return {};
}

std::expected<ScreenTiming, Refusal> PlaceboScreen::show(const ShowRequest& request) {
  Impl& s = *impl_;
  const pl_gpu gpu = s.vk->gpu;
  const std::int64_t start = host_now_us();
  OutputSize out = size();
  std::int64_t acquired = start;  // fin de l'acquisition de l'image de la swapchain
  pl_frame target{};
  if (s.offscreen) {
    auto drawn = s.offscreen_target();
    if (!drawn) return std::unexpected(std::move(drawn.error()));
    target = *drawn;
  } else {
    if (out != s.swapchain_size) {
      int w = out.width;
      int h = out.height;
      if (!pl_swapchain_resize(s.swapchain, &w, &h)) return ScreenTiming{.presented = false};
      s.swapchain_size = out;
    }
    pl_swapchain_frame swframe{};
    if (!pl_swapchain_start_frame(s.swapchain, &swframe))
      return ScreenTiming{.presented = false};  // fenêtre cachée
    pl_frame_from_swapchain(&target, &swframe);
    acquired = host_now_us();
  }
  s.hdr_output = target.color.transfer == PL_COLOR_TRC_PQ;
  s.target_color = target.color;
  s.take_next_grade(request.frame);
  s.select_grade(s.target_color);
  s.match_kwin();
  s.follow_ring();
  if (request.frame != nullptr && request.frame->pts_us != kNoPts && s.grade) {
    s.video_rate.observe(request.frame->pts_us, request.frame->epoch);
    s.grade->set_video_fps(s.video_rate.fps());
  }

  // La calibration du grade : la vidéo (celle de calibration, dans la même chaîne) à gauche du panneau, sans
  // sous-titres ni OSD ; l'interface reste dessus.
  const bool calibrating = request.calibration != nullptr;
  if (calibrating) {
    const CalibrationShot& place = *request.calibration;
    target.crop = {static_cast<float>(place.x0), static_cast<float>(place.y0), static_cast<float>(place.x1),
                   static_cast<float>(place.y1)};
  }
  const RingFrame* frame = request.frame;
  pl_frame image{};
  s.ring_image = false;
  if (frame != nullptr) {
    s.source_debanded = frame->debanded;
    s.shown_pts = frame->pts_us;
    s.shown_epoch = frame->epoch;
    auto tex = s.ring.acquire(gpu, *frame);
    if (!tex) return std::unexpected(std::move(tex.error()));
    image.num_planes = 1;
    image.planes[0].texture = *tex;
    image.planes[0].components = 3;
    image.planes[0].component_mapping[0] = 0;
    image.planes[0].component_mapping[1] = 1;
    image.planes[0].component_mapping[2] = 2;
    image.repr = pl_color_repr_rgb;
    image.color = s.source;
    // Gradée dans l'anneau : des codes PQ BT.2020 déjà faits pour la cible de leur version, montrés tels
    // quels.
    s.ring_image = frame->grade != 0;
    if (s.ring_image) {
      if (const pl_color_space* space = s.ring_space(frame->grade)) {
        image.color = *space;
      } else {
        image.color = s.target_color;
        image.color.primaries = PL_COLOR_PRIM_BT_2020;
        image.color.transfer = PL_COLOR_TRC_PQ;
      }
    }
    image.crop = {0, 0, static_cast<float>((*tex)->params.w), static_cast<float>((*tex)->params.h)};
    pl_rect2df_aspect_set(&target.crop, pl_rect2df_aspect(&image.crop) * static_cast<float>(s.sample_aspect),
                          0.0F);
  }

  auto overlays = s.overlays->build(
      calibrating ? nullptr : request.subtitles, calibrating || request.osd.empty() ? nullptr : &request.osd,
      *s.osd, out, request.ui != nullptr ? request.ui_bottom : 0, request.ui != nullptr ? request.ui_top : 0);
  if (!overlays) return std::unexpected(std::move(overlays.error()));
  target.overlays = overlays->all.data();
  target.num_overlays = static_cast<int>(overlays->all.size());

  // La composition gardée : seulement si d'autres présentations de la même image suivent, sans calibration
  // (son cadre change avec le panneau).
  bool kept = false;
  if (request.redraws_expected && !calibrating) {
    const ComposeState state{.crop = {target.crop.x0, target.crop.y0, target.crop.x1, target.crop.y1},
                             .grade_serial = s.grade_serial,
                             .final_scaler = s.final_scaler,
                             .deband = s.deband,
                             .sar = s.sample_aspect,
                             .transfer = static_cast<int>(target.color.transfer),
                             .primaries = static_cast<int>(target.color.primaries),
                             .max_luma = target.color.hdr.max_luma,
                             .calibrating = false};
    const ComposeKey key = compose_key(frame, request.subtitles, out.width, out.height,
                                       request.ui != nullptr ? request.ui_bottom : 0, state);
    auto done = s.compose_kept(frame != nullptr ? &image : nullptr, target, key, overlays->subtitles);
    if (!done) return std::unexpected(std::move(done.error()));
    kept = *done;
    if (kept) {
      if (auto drawn = s.draw_overlays(target, overlays->osd); !drawn)
        return std::unexpected(std::move(drawn.error()));
    }
  } else {
    s.composed_key.reset();
  }
  if (!kept) {
    if (auto drawn = s.compose(frame != nullptr ? &image : nullptr, target); !drawn) {
      return std::unexpected(std::move(drawn.error()));
    }
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
    if (auto saved = s.save_snapshot(frame != nullptr ? &image : nullptr, target, out, request.ui, path);
        !saved) {
      s.channel->warn("instantané {} : {}", path.string(), saved.error().detail);
    }
  }
  if (s.offscreen) {
    pl_gpu_flush(gpu);
    const std::int64_t submitted = host_now_us();
    const std::int64_t presented = s.offscreen_present();
    return ScreenTiming{.acquire_us = acquired - start,
                        .render_us = submitted - acquired,
                        .present_us = presented,
                        .queue_wait_us = presented - submitted,
                        .display_gpu_us = static_cast<std::int64_t>(s.display_frame_ns / 1000),
                        .presented = true};
  }
  // Retour de présentation (placebo_feedback.h) : ce qui est revenu est relu, la présentation part avec
  // son identifiant et sa demande de date.
  const pl_vulkan_present* present =
      s.feedback ? s.feedback->prepare(s.swapchain, host_now_us(), request.target_us) : nullptr;
  if (present != nullptr) {
    VkResult result = VK_SUCCESS;
    const bool ok = pl_vulkan_swapchain_submit(s.swapchain, present, &result);
    s.feedback->submitted(result);
    if (!ok) return refuse(RefusalCode::VendorCallFailed, "pl_vulkan_swapchain_submit");
  } else if (!pl_swapchain_submit_frame(s.swapchain)) {
    return refuse(RefusalCode::VendorCallFailed, "pl_swapchain_submit_frame");
  }
  const std::int64_t submitted = host_now_us();
  pl_swapchain_swap_buffers(s.swapchain);  // FIFO : attend une place en file quand elle est pleine
  const std::int64_t presented = host_now_us();
  ScreenTiming timing{.acquire_us = acquired - start,
                      .render_us = submitted - acquired,
                      .present_us = presented,
                      .queue_wait_us = presented - submitted,
                      .display_gpu_us = static_cast<std::int64_t>(s.display_frame_ns / 1000),
                      .presented = true};
  if (s.feedback) {
    const FeedbackRead& read = s.feedback->last_read();
    timing.present_id = s.feedback->last_id();
    timing.level = s.feedback->level();
    timing.feedback_id = read.id;
    timing.target_us = read.target_us;
    timing.visible_us = read.visible_us;
    timing.vsync_k = read.vsync_k;
  }
  return timing;
}

}  // namespace ajp::app::play
