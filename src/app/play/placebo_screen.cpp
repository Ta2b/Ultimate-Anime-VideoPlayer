#include "app/play/placebo_screen.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <mutex>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_filesystem.h>
#include <SDL3/SDL_video.h>
#include <SDL3/SDL_vulkan.h>
#include <libplacebo/log.h>
#include <libplacebo/renderer.h>
#include <libplacebo/swapchain.h>
#include <libplacebo/vulkan.h>

#include "app/play/i18n.h"
#include "app/play/osd_text.h"
#include "app/play/placebo_impl.h"
#include "core/clock.h"
#include "core/log.h"
#include "core/trace.h"
#include "core/utf8_path.h"

namespace ajp::app::play {
namespace {
// Observations de la grille des vblanks avant de s'y fier pour la synchro (≈ 0,7 s à 165 Hz).
constexpr std::size_t kGridSamples = 120;
}  // namespace
namespace {

// Blanc de l'interface dans une sortie HDR : le blanc de référence (BT.2408), comme le SDR.
constexpr float kUiWhiteNits = 203.0F;

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
  return impl_->hdr_output ? i18n::tr("HDR10 output (PQ BT.2020)") : i18n::tr("SDR output");
}

bool PlaceboScreen::has_grade() const {
  return impl_->grade.has_value();
}

std::expected<std::vector<RingSlotDesc>, Refusal> PlaceboScreen::create_ring(int width, int height,
                                                                             std::uint32_t count,
                                                                             bool external,
                                                                             RingFormat format) {
  return impl_->ring.create(impl_->vk->gpu, width, height, count, external, format);
}

void PlaceboScreen::release_ring() {
  Impl& s = *impl_;
  s.ring.clear(s.vk->gpu);
  pl_tex_destroy(s.vk->gpu, &s.composed);
  s.composed_key.reset();
  if (s.renderer != nullptr) pl_renderer_flush_cache(s.renderer);
}

bool PlaceboScreen::hdr_output() const {
  return impl_->hdr_output;
}

void PlaceboScreen::set_sdr_output(bool sdr) {
  Impl& s = *impl_;
  if (sdr == s.sdr_forced) return;
  s.sdr_forced = sdr;
  s.hdr_wanted = s.hdr_allowed && !sdr;
  s.channel->info("sortie demandée : {}", sdr || !s.hdr_allowed ? "SDR" : "HDR10");  // i18n-ok : journal
  if (s.swapchain == nullptr) return;
  // Les métadonnées annoncées repartent de zéro : au retour en HDR, match_kwin les repose.
  s.hinted_max_nits = 0.0;
  s.hinted_max_fall = 0.0;
  pl_swapchain_colorspace_hint(s.swapchain, s.hdr_wanted ? &pl_color_space_hdr10 : nullptr);
  s.composed_key.reset();
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

void PlaceboScreen::Impl::draw_ui(const ImDrawData* data, const pl_frame& target, std::uint64_t serial) {
  if (data == nullptr && ui) ui->drop_cache();  // interface cachée : sa texture gardée est rendue
  if (data == nullptr || ui_failed) return;
  if (auto drawn = ui->draw(*data, target, kUiWhiteNits, serial); !drawn) {
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

// Anime4K sur MAIN d'abord (l'agrandissement, render/anime4k.h), la netteté sur SCALED ensuite, puis les
// passes du grade sur OUTPUT. Une source SDR arrive à la netteté en lumière linéaire, une source HDR encodée
// (A41-06).
std::size_t PlaceboScreen::Impl::video_hooks(std::array<const pl_hook*, 16>& hooks, const pl_frame* image,
                                             bool graded_now) {
  std::size_t count = 0;
  const auto add = [&](std::span<const pl_hook* const> some) {
    for (const pl_hook* hook : some) {
      if (count < hooks.size()) hooks[count++] = hook;
    }
  };
  if (anime4k && image != nullptr) add(anime4k->hooks());
  const bool sdr_source = image == nullptr || !pl_color_space_is_hdr(&image->color);
  if (const pl_hook* sharp = sharpen ? sharpen->hook_for(sdr_source) : nullptr; sharp != nullptr)
    hooks[count++] = sharp;
  if (graded_now && grade) add(grade->hooks());
  return count;
}

void PlaceboScreen::Impl::note_pass(const pl_dispatch_info& pass) const {
  display_frame_ns += pass.last;
  const char* what = pass.shader->description;
  if (what == nullptr) return;
  if (std::string_view(what).find("AnimeJaNai") != std::string_view::npos) grade_frame_ns += pass.last;
  if (render::is_anime4k_pass(what)) anime4k_frame_ns += pass.last;
}

void PlaceboScreen::Impl::keep_pass_times(bool graded_now) const {
  if (graded_now && grade_frame_ns > 0) grade_gpu_us.add(static_cast<std::uint32_t>(grade_frame_ns / 1000));
  if (anime4k && anime4k_frame_ns > 0)
    anime4k_gpu_us.add(static_cast<std::uint32_t>(anime4k_frame_ns / 1000));
}

std::expected<void, Refusal> PlaceboScreen::Impl::compose(const pl_frame* image, const pl_frame& target) {
  AJP_TRACE("composition");
  pl_render_params params = pl_render_default_params;
  switch (final_scaler) {
    case FinalScaler::Default:
      break;
    // Le même filtre en réduction (P3.68) : une chaîne plus grande que la fenêtre gardait celui par défaut.
    case FinalScaler::EwaLanczos:
      params.upscaler = params.downscaler = &pl_filter_ewa_lanczos;
      break;
    case FinalScaler::EwaLanczosSharp:
      params.upscaler = params.downscaler = &pl_filter_ewa_lanczossharp;
      break;
  }
  // Le grade chargé est celui du PQ ou du SDR : sur une autre cible (instantané SDR d'une sortie
  // HDR10), il est sauté.
  const bool graded_now = grade.has_value() && graded(target.color);
  // La case Débandage décide sur tous les moteurs (P3.56) : une image déjà débandée avant SPAN (TensorRT) ne
  // l'est pas deux fois ; sinon libplacebo débande ici. Sous le grade, son grain fabriquait du bruit dans les
  // aplats sombres que le grade étire (écart-type 16x16 de 1,0 contre 0,0) : il y débande sans grain.
  static const pl_deband_params kQuietDeband = [] {
    pl_deband_params quiet = pl_deband_default_params;
    quiet.grain = 0.0F;
    return quiet;
  }();
  if (deband && !frame_debanded)
    params.deband_params = graded_now ? &kQuietDeband : &pl_deband_default_params;
  std::array<const pl_hook*, 16> hooks{};
  if (const std::size_t count = video_hooks(hooks, image, graded_now); count > 0) {
    params.hooks = hooks.data();
    params.num_hooks = static_cast<int>(count);
  }
  // Le temps GPU de chaque passe, toujours relevé (libplacebo le rend avec un retard d'une image ou deux) :
  // toutes les passes pour l'affichage, celles du grade et d'Anime4K à part.
  params.info_callback = [](void* priv, const pl_render_info* info) {
    static_cast<Impl*>(priv)->note_pass(*info->pass);
  };
  params.info_priv = const_cast<Impl*>(this);
  grade_frame_ns = 0;
  anime4k_frame_ns = 0;
  display_frame_ns = 0;
  if (image == nullptr) pl_frame_clear(vk->gpu, &target, std::array<float, 3>{0.0F, 0.0F, 0.0F}.data());
  if (!pl_render_image(renderer, image, &target, &params)) {
    return std::unexpected(Refusal{RefusalCode::VendorCallFailed, "pl_render_image a échoué"});
  }
  if (image != nullptr) keep_pass_times(graded_now);
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

std::optional<Refusal> PlaceboScreen::device_lost() const {
  const Impl& s = *impl_;
  if (s.ring.lost() || (s.vk != nullptr && pl_gpu_is_failed(s.vk->gpu)))
    return Refusal{RefusalCode::DeviceLost, "appareil Vulkan de l'écran perdu (TDR, pilote rechargé)"};
  return std::nullopt;
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
  return impl_->watcher != nullptr && impl_->watcher->alive();
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
    params.blit_dst = true;  // le fond noir d'une présentation sans image s'écrit par un effacement
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
  // Sans image, libplacebo efface la cible avant de dessiner : ici elle porte déjà la composition recopiée.
  pl_render_params params = pl_render_default_params;
  params.border = PL_CLEAR_SKIP;
  if (!pl_render_image(renderer, nullptr, &on, &params))
    return std::unexpected(Refusal{RefusalCode::VendorCallFailed, "pl_render_image (OSD) a échoué"});
  return {};
}

namespace {

// L'acquisition (de asked à acquired) a attendu l'écran : en FIFO, une image vient d'être rendue à un vsync,
// son retour le date (A12-03). Sinon 0 : cette présentation ne dit rien de la phase de l'écran.
std::int64_t blocked_vsync(std::int64_t asked, std::int64_t acquired) {
  return acquired - asked >= kAcquireBlockedUs ? acquired : 0;
}

// Le retour de présentation relu avant celle-ci, et son identifiant.
void note_feedback(ScreenTiming& timing, const PresentFeedback& feedback) {
  const FeedbackRead& read = feedback.last_read();
  timing.present_id = feedback.last_id();
  timing.level = feedback.level();
  timing.feedback_id = read.id;
  timing.target_us = read.target_us;
  timing.visible_us = read.visible_us;
  timing.vsync_k = read.vsync_k;
}

}  // namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
// La présentation part. Aucun autre thread n'ouvre de commande sur le pl_gpu pendant ce temps (une commande
// ouverte entre la prise de la swapchain et sa soumission fait échouer vk_cmd_submit dans libplacebo).
std::expected<void, Refusal> PlaceboScreen::Impl::submit(const pl_vulkan_present* present) {
  frame_pending = false;
  if (present == nullptr) {
    if (pl_swapchain_submit_frame(swapchain)) return {};
    return refuse(RefusalCode::VendorCallFailed, "pl_swapchain_submit_frame");
  }
  VkResult result = VK_SUCCESS;
  const bool ok = pl_vulkan_swapchain_submit(swapchain, present, &result);
  feedback->submitted(result);
  if (!ok) return refuse(RefusalCode::VendorCallFailed, "pl_vulkan_swapchain_submit");
  return {};
}

bool PlaceboScreen::Impl::start_frame(pl_swapchain_frame* frame) {
  if (frame_pending) {
    channel->warn(
        "présentation : l'image acquise au tour précédent n'avait pas été soumise, elle part d'abord");
    if (auto sent = submit(nullptr); !sent) channel->warn("présentation : {}", sent.error().detail);
    pl_swapchain_swap_buffers(swapchain);
  }
  frame_pending = pl_swapchain_start_frame(swapchain, frame);
  return frame_pending;
}

// L'image de la swapchain où présenter, et la taille qu'elle a vraiment ; faux : rien n'est présenté (fenêtre
// cachée, swapchain à refaire).
std::expected<bool, Refusal> PlaceboScreen::acquire(OutputSize& out, pl_frame& target) {
  Impl& s = *impl_;
  // Fenêtre réduite ou que le compositeur ne montre plus (Wayland « suspended », macOS) : rien n'est
  // présenté (P2.19). En FIFO, la présentation y attendrait un rappel de rafraîchissement qui ne vient pas,
  // et le thread de la fenêtre resterait bloqué. L'attente est rendue au premier événement, 50 ms au plus.
  if ((SDL_GetWindowFlags(s.window) & (SDL_WINDOW_OCCLUDED | SDL_WINDOW_MINIMIZED)) != 0) {
    constexpr Sint32 kHiddenWaitMs = 50;
    (void)SDL_WaitEventTimeout(nullptr, kHiddenWaitMs);
    s.acquire_streak.reset();
    return false;
  }
  // A10-01 : un échec d'acquisition sur un appareil en échec est une perte du GPU, pas une fenêtre cachée.
  // B31-15 : sur un appareil sain, des échecs qui durent sont une surface perdue ; la relance la recrée.
  const auto not_shown = [&]() -> std::expected<bool, Refusal> {
    if (auto lost = device_lost()) return std::unexpected(std::move(*lost));
    if (s.acquire_streak.failed(host_now_us())) {
      s.channel->error("présentation : aucune image acquise depuis {} s sur une fenêtre visible",
                       kLostSurfaceUs / 1'000'000);
      return std::unexpected(Refusal{RefusalCode::DeviceLost,
                                     "surface de la fenêtre perdue (écran débranché ou sortie détruite)"});
    }
    return false;
  };
  if (out != s.swapchain_size) {
    int w = out.width;
    int h = out.height;
    if (!pl_swapchain_resize(s.swapchain, &w, &h)) return not_shown();
    s.swapchain_size = out;
  }
  pl_swapchain_frame swframe{};
  if (!s.start_frame(&swframe)) return not_shown();  // fenêtre cachée, surface perdue ou GPU perdu
  s.acquire_streak.reset();
  pl_frame_from_swapchain(&target, &swframe);
  // B10-05 : la taille réellement prise par la swapchain (currentExtent imposé par la surface pendant un
  // redimensionnement ou un changement d'échelle), pas celle demandée : sous-titres, OSD et interface
  // se placent dans la cible de cette présentation.
  if (swframe.fbo != nullptr) out = OutputSize{swframe.fbo->params.w, swframe.fbo->params.h};
  return true;
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
    auto shown = acquire(out, target);
    if (!shown) return std::unexpected(std::move(shown.error()));
    if (!*shown) return ScreenTiming{.presented = false};
    acquired = host_now_us();
  }
  s.hdr_output = target.color.transfer == PL_COLOR_TRC_PQ;
  s.target_color = target.color;
  s.take_next_grade(request.frame);
  s.select_grade(s.target_color);
  s.match_kwin();
  if (request.frame != nullptr && request.frame->pts_us != kNoPts && s.grade) {
    s.video_rate.observe(request.frame->pts_us, request.frame->epoch);
    s.grade->set_video_fps(s.video_rate.fps());
  }

  const RingFrame* frame = request.frame;
  s.frame_debanded = frame != nullptr && frame->debanded;
  pl_frame image{};
  if (frame != nullptr) {
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
    image.crop = {0, 0, static_cast<float>((*tex)->params.w), static_cast<float>((*tex)->params.h)};
    pl_rect2df_aspect_set(&target.crop, pl_rect2df_aspect(&image.crop) * static_cast<float>(s.sample_aspect),
                          0.0F);
  }

  auto overlays = s.overlays->build(request.subtitles, request.osd.empty() ? nullptr : &request.osd, *s.osd,
                                    out, request.ui != nullptr ? request.ui_bottom : 0,
                                    request.ui != nullptr ? request.ui_top : 0, request.tint);
  if (!overlays) return std::unexpected(std::move(overlays.error()));
  target.overlays = overlays->all.data();
  target.num_overlays = static_cast<int>(overlays->all.size());

  // La composition gardée : seulement si d'autres présentations de la même image suivent.
  bool kept = false;
  if (request.redraws_expected) {
    const ComposeState state{.crop = {target.crop.x0, target.crop.y0, target.crop.x1, target.crop.y1},
                             .grade_serial = s.grade_serial,
                             .final_scaler = s.final_scaler,
                             .deband = s.deband,
                             .sharpness = s.sharpness,
                             .anime4k = s.anime4k_key(),
                             .sar = s.sample_aspect,
                             .transfer = static_cast<int>(target.color.transfer),
                             .primaries = static_cast<int>(target.color.primaries),
                             .max_luma = target.color.hdr.max_luma};
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
  s.draw_ui(request.ui, target, request.ui_serial);
  if (s.snapshot) s.take_snapshot(frame != nullptr ? &image : nullptr, target, out, request.ui);
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
  if (auto sent = s.submit(present); !sent) return std::unexpected(std::move(sent.error()));
  const std::int64_t submitted = host_now_us();
  pl_swapchain_swap_buffers(s.swapchain);  // attend les commandes des images en vol, pas l'écran (A10-14)
  const std::int64_t presented = host_now_us();
  ScreenTiming timing{.acquire_us = acquired - start,
                      .render_us = submitted - acquired,
                      .present_us = presented,
                      .queue_wait_us = presented - submitted,
                      .vsync_us = blocked_vsync(start, acquired),
                      .display_gpu_us = static_cast<std::int64_t>(s.display_frame_ns / 1000),
                      .presented = true};
  if (s.feedback) note_feedback(timing, *s.feedback);
  return timing;
}

}  // namespace ajp::app::play
