#include "render/compositor.h"

#include <array>
#include <cstddef>
#include <utility>

#include <libplacebo/renderer.h>

#include "render/grade.h"
#include "render/imgui_layer.h"
#include "render/log.h"
#include "render/render_device.h"
#include "render/subtitle_layer.h"

namespace ajp::render {
namespace {

// Le slot entier, RGB plein sur 16 bits ; son alpha n'a pas de sens et n'est pas lu.
pl_frame slot_frame(pl_tex slot, const pl_color_space& color) {
  pl_frame image{};
  image.num_planes = 1;
  image.planes[0].texture = slot;
  image.planes[0].components = 3;
  image.planes[0].component_mapping[0] = PL_CHANNEL_R;
  image.planes[0].component_mapping[1] = PL_CHANNEL_G;
  image.planes[0].component_mapping[2] = PL_CHANNEL_B;
  image.planes[0].component_mapping[3] = PL_CHANNEL_NONE;
  image.repr = pl_color_repr_rgb;
  image.repr.bits = {.sample_depth = 16, .color_depth = 16};
  image.color = color;
  image.crop = {.x0 = 0.0F,
                .y0 = 0.0F,
                .x1 = static_cast<float>(slot->params.w),
                .y1 = static_cast<float>(slot->params.h)};
  return image;
}

// Le rectangle de l'image dans la cible : tout le crop demandé (ou toute la cible), réduit au
// rapport d'aspect du slot et centré ; le reste devient des bandes noires.
pl_rect2df letterbox(const pl_frame& target, const pl_rect2df& source) {
  pl_rect2df dst = target.crop;
  if (dst.x0 == dst.x1 || dst.y0 == dst.y1) {
    const pl_tex tex = target.planes[0].texture;
    dst = {.x0 = 0.0F,
           .y0 = 0.0F,
           .x1 = static_cast<float>(tex->params.w),
           .y1 = static_cast<float>(tex->params.h)};
  }
  pl_rect2df_aspect_copy(&dst, &source, 0.0F);
  return dst;
}

}  // namespace

struct Compositor::Impl {
  std::shared_ptr<spdlog::logger> logger = make_logger();
  pl_gpu gpu = nullptr;
  pl_renderer renderer = nullptr;
  Grade grade;
  SubtitleLayer subtitles;
  ImGuiLayer osd;
  CompositionSettings settings;
  double peak_nits = grade_limits::kDefaultPeakNits;
  bool grade_enabled = true;
  GradeMode mode = GradeMode::Neutral;
  std::uint64_t gpu_time_ns = 0;
  std::uint64_t frame_gpu_time_ns = 0;

  Impl(pl_gpu gpu_in, pl_log log, Grade grade_in)
      : gpu(gpu_in),
        renderer(pl_renderer_create(log, gpu_in)),
        grade(std::move(grade_in)),
        subtitles(gpu_in),
        osd(gpu_in, log) {}
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;
  ~Impl() { pl_renderer_destroy(&renderer); }

  static void on_pass(void* priv, const pl_render_info* info) {
    static_cast<Impl*>(priv)->frame_gpu_time_ns += info->pass->last;
  }
};

std::expected<Compositor, Refusal> Compositor::create(const RenderDevice& device) {
  auto grade = Grade::create(device.gpu());
  if (!grade) return std::unexpected(grade.error());
  auto impl = std::make_unique<Impl>(device.gpu(), device.log(), std::move(*grade));
  if (impl->renderer == nullptr)
    return std::unexpected(Refusal{RefusalCode::RenderFailed, "pl_renderer_create"});
  return Compositor(std::move(impl));
}

Compositor::Compositor(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Compositor::Compositor(Compositor&&) noexcept = default;
Compositor& Compositor::operator=(Compositor&&) noexcept = default;
Compositor::~Compositor() = default;

std::expected<void, Refusal> Compositor::configure(const CompositionSettings& settings) {
  if (auto ok = check_grade_settings(settings.grade); !ok) return ok;
  if (auto ok = core::check_display(settings.display); !ok) return ok;
  auto peak = grade_peak(settings.display);
  if (!peak) return std::unexpected(peak.error());
  impl_->settings = settings;
  impl_->peak_nits = *peak;
  return {};
}

const CompositionSettings& Compositor::settings() const {
  return impl_->settings;
}

void Compositor::set_grade_enabled(bool enabled) {
  impl_->grade_enabled = enabled;
}

bool Compositor::grade_enabled() const {
  return impl_->grade_enabled;
}

GradeMode Compositor::grade_mode() const {
  return impl_->mode;
}

std::expected<void, Refusal> Compositor::compose(const pl_tex_t* slot, const SlotColor& color,
                                                 const pl_frame& target, const ComposeLayers& layers) {
  Impl& s = *impl_;
  pl_frame out = target;
  if (out.color.transfer == PL_COLOR_TRC_UNKNOWN) out.color = detail::to_placebo(s.settings.display);
  const bool output_hdr = pl_color_space_is_hdr(&out.color);
  const bool output_pq = out.color.transfer == PL_COLOR_TRC_PQ;

  const GradeContext context{.route = s.settings.route,
                             .source_hdr = is_hdr(color.transfer),
                             .output_pq = output_pq,
                             .enabled = s.grade_enabled,
                             .peak_nits = s.peak_nits};
  auto decision = decide_grade(s.settings.grade, context);
  if (!decision) return std::unexpected(decision.error());
  s.mode = decision->mode;

  const pl_frame image =
      slot_frame(slot, detail::to_placebo(color, output_hdr ? decision->source_white_nits : 0.0));
  out.crop = letterbox(out, image.crop);

  // Sans allocation : au plus les deux hooks du grade, ou la mire seule.
  std::array<const pl_hook*, 2> hooks{};
  std::size_t num_hooks = 0;
  if (layers.calibration && output_pq) {
    hooks[num_hooks++] = s.grade.calibration();
  } else if (s.mode != GradeMode::Neutral) {
    if (s.grade.params() != decision->params) s.grade.set_params(decision->params);
    for (const pl_hook* hook : s.grade.hooks()) hooks.at(num_hooks++) = hook;
  }

  if (layers.subtitles != nullptr) {
    const auto white = output_hdr ? static_cast<float>(s.settings.grade.subtitle_white) : 0.0F;
    auto overlays = s.subtitles.prepare(*layers.subtitles, out.crop, white);
    if (!overlays) return std::unexpected(overlays.error());
    out.overlays = overlays->data();
    out.num_overlays = static_cast<int>(overlays->size());
  }

  pl_render_params params = pl_render_default_params;
  // Ni mesure de crête ni courbe dynamique : TrueHDR arrive déjà au pic de l'écran, le grade
  // mesure ses scènes lui-même, et le rendu d'un slot reste déterministe.
  params.peak_detect_params = nullptr;
  params.hooks = hooks.data();
  params.num_hooks = static_cast<int>(num_hooks);
  params.info_callback = &Impl::on_pass;
  params.info_priv = &s;
  s.frame_gpu_time_ns = 0;
  if (!pl_render_image(s.renderer, &image, &out, &params)) {
    return std::unexpected(Refusal{RefusalCode::RenderFailed, "pl_render_image (voir le journal render)"});
  }
  s.gpu_time_ns = s.frame_gpu_time_ns;

  if (layers.osd != nullptr) {
    out.overlays = nullptr;
    out.num_overlays = 0;
    const auto white = static_cast<float>(s.settings.display.sdr_white_nits);
    if (auto ok = s.osd.draw(*layers.osd, out, white); !ok) return ok;
  }
  return {};
}

void Compositor::reset() {
  pl_renderer_flush_cache(impl_->renderer);
}

void Compositor::drop_subtitles() {
  impl_->subtitles.drop_atlas();
}

std::uint64_t Compositor::last_gpu_time_ns() const {
  return impl_->gpu_time_ns;
}

void Compositor::configure_imgui(ImGuiIO& io) {
  ImGuiLayer::configure(io);
}

void Compositor::release_imgui_textures(ImGuiPlatformIO& platform) {
  impl_->osd.release_textures(platform);
}

}  // namespace ajp::render
