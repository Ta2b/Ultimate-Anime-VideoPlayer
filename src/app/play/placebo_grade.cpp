// Couleurs et grade de l'écran libplacebo : l'espace de la source, le grade de chaque cible (PQ ou
// SDR à son exposant) et la mesure de la rampe du vrai noir. Ouverture dans placebo_open.cpp,
// présentation dans placebo_screen.cpp.
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
#include "app/play/placebo_screen.h"
#include "core/clock.h"
#include "core/log.h"
#include "kernels/spec/grade.h"
#include "render/display_model.h"

namespace ajp::app::play {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

// Nits d'un code PQ (ST 2084).
double pq_nits(double e) {
  const double p = std::pow(std::clamp(e, 0.0, 1.0), 1.0 / 78.84375);
  return 10000.0 * std::pow(std::max(p - 0.8359375, 0.0) / (18.8515625 - 18.6875 * p), 1.0 / 0.1593017578125);
}

// L'exposant du transfert d'une cible SDR BT.709 ; rien si le grade ne sait pas y travailler.
std::optional<double> sdr_gamma(const pl_color_space& color) {
  if (color.primaries != PL_COLOR_PRIM_BT_709) return std::nullopt;
  switch (color.transfer) {
    case PL_COLOR_TRC_SRGB:
    case PL_COLOR_TRC_GAMMA22:
      return 2.2;
    case PL_COLOR_TRC_BT_1886:
    case PL_COLOR_TRC_GAMMA24:
      return 2.4;
    default:
      return std::nullopt;
  }
}

// Le grade d'une cible : celui du PQ, ou celui du SDR à l'exposant de la cible.
std::optional<render::GradeParams> grade_for(const render::GradeChoice& choice, const pl_color_space& color) {
  if (color.transfer == PL_COLOR_TRC_PQ) return choice.pq;
  const std::optional<double> gamma = sdr_gamma(color);
  if (!gamma || !choice.sdr) return std::nullopt;
  render::GradeParams params = *choice.sdr;
  params.sdr_gamma = *gamma;
  return params;
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
  if (color.hdr) {  // crête et lumière de la source (TrueHDR : le pic qu'on lui a donné)
    space.hdr.max_luma = color.hdr->max_luminance;
    space.hdr.min_luma = color.hdr->min_luminance;
    space.hdr.max_cll = color.hdr->max_cll;
    space.hdr.max_fall = color.hdr->max_fall;
  }
  return space;
}

}  // namespace

void PlaceboScreen::set_source_color(const ColorInfo& color) {
  const pl_color_space space = color_space_of(color);
  if (!pl_color_space_equal(&space, &impl_->source)) ++impl_->grade_serial;
  impl_->source = space;
}

void PlaceboScreen::set_sample_aspect(double sar) {
  impl_->sample_aspect = std::isfinite(sar) && sar > 0.0 ? sar : 1.0;
}

void PlaceboScreen::set_finish(FinalScaler scaler, bool deband) {
  impl_->final_scaler = scaler;
  impl_->deband = deband;
}

void PlaceboScreen::set_grade(const render::GradeChoice& choice) {
  impl_->grade_next.reset();
  impl_->ring_ahead = false;
  impl_->grade_choice = choice;
  impl_->select_grade(impl_->target_color);
}

void PlaceboScreen::set_grade_after(const render::GradeChoice& choice, const RingFrame& after) {
  if (after.pts_us == kNoPts) {
    set_grade(choice);
    return;
  }
  Impl& s = *impl_;
  if (!s.grade_next) s.grade_next_since_us = host_now_us();  // une suite de changements garde son départ
  s.grade_next = choice;
  s.grade_next_pts = after.pts_us;
  s.grade_next_epoch = after.epoch;
  // L'anneau la reçoit tout de suite, pour les images qui suivent `after` : celles que l'écran montrera avec
  // ce choix.
  if (s.grade_ring && s.ring_live) {
    s.post_ring(choice, after.pts_us + 1, after.epoch);
    s.ring_ahead = true;
  }
}

void PlaceboScreen::Impl::take_next_grade(const RingFrame* frame) {
  if (!grade_next) return;
  constexpr std::int64_t kWaitUs = 250'000;
  const bool due = frame == nullptr || frame->epoch != grade_next_epoch || frame->pts_us == kNoPts ||
                   frame->pts_us > grade_next_pts || host_now_us() - grade_next_since_us > kWaitUs;
  if (!due) return;
  grade_choice = *grade_next;
  grade_next.reset();
  ring_absorb = std::exchange(ring_ahead, false);
}

std::expected<void, Refusal> PlaceboScreen::set_looks(
    const std::array<const render::CubeLut*, render::kLookSlots>& luts) {
  if (!impl_->grade) return refuse(RefusalCode::HdrUnavailable, "look : pas de grade sur cette sortie");
  if (auto set = impl_->grade->set_looks(luts); !set) return set;
  ++impl_->grade_serial;
  impl_->select_grade(impl_->target_color);  // les paramètres du look repartent dans le nouveau hook
  return {};
}

void PlaceboScreen::Impl::select_grade(const pl_color_space& target) {
  const std::optional<render::GradeParams> params = grade_for(grade_choice, target);
  const std::optional<bool> before = grade_loaded_pq;
  grade_loaded_pq.reset();
  if (grade && params) {
    if (*params != grade->params()) {
      grade->set_params(*params);
      ++grade_serial;
    }
    grade_loaded_pq = target.transfer == PL_COLOR_TRC_PQ;
  }
  if (grade_loaded_pq != before) ++grade_serial;
}

void PlaceboScreen::Impl::match_kwin() {
  if (!kde_sdr_nits || !grade || swapchain == nullptr) return;
  const bool on = hdr_output && grade_loaded_pq == true;
  // Le pic du grade est en unités du contenu ; la passe écrête au pic du panneau, et la sortie annonce celui
  // du contenu : KWin le ramène à son blanc (render::kwin_hint).
  const double peak = grade->params().peak;
  const render::KwinHint hint = render::kwin_hint(peak, *kde_sdr_nits);
  const std::optional<render::Grade::KwinMatch> match =
      on ? std::optional(render::Grade::KwinMatch{.reference = *kde_sdr_nits, .peak = hint.panel_peak_nits})
         : std::nullopt;
  if (match != kwin_applied) ++grade_serial;
  kwin_applied = match;
  grade->set_kwin_match(match);
  const double max_nits = on ? hint.max_content_nits : 0.0;
  if (max_nits == hinted_max_nits) return;
  hinted_max_nits = max_nits;
  pl_color_space space = pl_color_space_hdr10;
  if (on) {
    space.hdr.max_luma = static_cast<float>(max_nits);
    space.hdr.min_luma = 0.0001F;
    space.hdr.max_cll = static_cast<float>(max_nits);
  }
  pl_swapchain_colorspace_hint(swapchain, &space);
  channel->info("--hdr-metadata : maximum annoncé {:.0f} nits (pic {:.0f}, luminosité SDR de KDE {:.0f})",
                max_nits, peak, *kde_sdr_nits);
}

bool PlaceboScreen::Impl::graded(const pl_color_space& target) const {
  const bool pq = target.transfer == PL_COLOR_TRC_PQ;
  return grade && grade_loaded_pq == pq && (pq || sdr_gamma(target).has_value());
}

std::expected<std::vector<double>, Refusal> PlaceboScreen::grade_ramp(std::span<const float> levels) {
  Impl& s = *impl_;
  if (!s.grade || !s.graded(s.target_color))
    return refuse(RefusalCode::HdrUnavailable, "rampe du vrai noir : pas de grade sur cette sortie");
  const pl_gpu gpu = s.vk->gpu;
  if (s.ramp_renderer == nullptr) s.ramp_renderer = pl_renderer_create(s.log, gpu);
  if (!s.ramp_grade) {
    auto grade = render::Grade::create(gpu, s.grade_body, s.skin_mask);
    if (!grade) return std::unexpected(std::move(grade.error()));
    s.ramp_grade.emplace(std::move(*grade));
  }
  render::GradeParams params = s.grade->params();
  params.black_lo = 0.0;  // le vrai noir se cale sur ce qui sort sans lui
  params.black_hi = 0.0;
  if (s.last_ramp && s.last_ramp->params == params && s.last_ramp->hdr == s.hdr_output &&
      std::ranges::equal(s.last_ramp->levels, levels))
    return s.last_ramp->nits;
  s.ramp_grade->set_params(params);
  // Une bande de 16 px par marche, 128 lignes : assez pour les cartes 1/8 et 1/32 du grade.
  constexpr int kBand = 16;
  constexpr int kRows = 128;
  const int width = kBand * static_cast<int>(levels.size());
  std::vector<std::uint16_t> pixels(static_cast<std::size_t>(width) * kRows * 4);
  for (int y = 0; y < kRows; ++y) {
    for (int x = 0; x < width; ++x) {
      const auto value = static_cast<std::uint16_t>(
          std::lround(std::clamp(levels[static_cast<std::size_t>(x / kBand)], 0.0F, 1.0F) * 65535.0F));
      const std::size_t at =
          (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)) * 4;
      pixels[at] = pixels[at + 1] = pixels[at + 2] = value;
      pixels[at + 3] = 65535;
    }
  }
  pl_tex_params tex{};
  tex.w = width;
  tex.h = kRows;
  tex.format = pl_find_named_fmt(gpu, "rgba16");
  tex.sampleable = true;
  tex.host_writable = true;
  tex.initial_data = pixels.data();
  pl_tex in = pl_tex_create(gpu, &tex);
  tex.initial_data = nullptr;
  tex.sampleable = false;
  tex.host_writable = false;
  tex.renderable = true;
  tex.host_readable = true;
  pl_tex out = pl_tex_create(gpu, &tex);
  const auto frame_of = [&](pl_tex texture, const pl_color_space& color) {
    pl_frame frame{};
    frame.num_planes = 1;
    frame.planes[0].texture = texture;
    frame.planes[0].components = 4;
    for (int c = 0; c < 4; ++c) frame.planes[0].component_mapping[c] = c;
    frame.repr = pl_color_repr_rgb;
    frame.color = color;
    frame.crop = {0, 0, static_cast<float>(width), static_cast<float>(kRows)};
    return frame;
  };
  bool done = in != nullptr && out != nullptr;
  if (done) {
    const pl_frame image = frame_of(in, s.source);  // comme une image du ring
    const pl_frame target = frame_of(out, s.target_color);
    pl_render_params render = pl_render_default_params;
    render.dither_params = nullptr;
    const std::span<const pl_hook* const> hooks = s.ramp_grade->hooks();
    render.hooks = hooks.data();
    render.num_hooks = static_cast<int>(hooks.size());
    done = pl_render_image(s.ramp_renderer, &image, &target, &render);
    pl_tex_transfer_params download{};
    download.tex = out;
    download.ptr = pixels.data();
    done = done && pl_tex_download(gpu, &download);
  }
  pl_tex_destroy(gpu, &in);
  pl_tex_destroy(gpu, &out);
  if (!done) return refuse(RefusalCode::VendorCallFailed, "rampe du vrai noir : rendu ou relecture");
  std::vector<double> nits;
  nits.reserve(levels.size());
  const std::size_t middle = static_cast<std::size_t>(kRows / 2) * static_cast<std::size_t>(width);
  for (std::size_t i = 0; i < levels.size(); ++i) {
    const std::size_t at = (middle + i * kBand + kBand / 2) * 4;
    const std::uint16_t peak = std::max({pixels[at], pixels[at + 1], pixels[at + 2]});
    const double code = peak / 65535.0;
    nits.push_back(s.hdr_output ? pq_nits(code) : std::pow(code, params.sdr_gamma) * params.peak);
  }
  s.last_ramp = Impl::RampMeasure{
      .params = params, .levels = {levels.begin(), levels.end()}, .hdr = s.hdr_output, .nits = nits};
  return nits;
}

// Le grade de l'anneau (chantier 2c, docs/adr/0009-grade-dans-l-anneau.md) : les versions postées dans les
// réglages en direct, que le backend du ring applique entre l'écriture d'un emplacement et son signal.

void PlaceboScreen::set_ring_live(std::shared_ptr<LiveSettings> live) {
  impl_->ring_live = std::move(live);
}

bool PlaceboScreen::stale_grade(const RingFrame& frame) const {
  const Impl& s = *impl_;
  if (frame.grade == 0 || !s.ring_live) return false;
  const std::shared_ptr<const GradeVersion> wanted = s.ring_live->grade_at(frame.pts_us, frame.epoch);
  const std::uint32_t expected = wanted && !wanted->off ? wanted->version : 0;
  return frame.grade != expected;
}

const pl_color_space* PlaceboScreen::Impl::ring_space(std::uint32_t version) const {
  for (const RingVersion& v : ring_versions) {
    if (v.version == version && version != 0) return &v.space;
  }
  return nullptr;
}

void PlaceboScreen::Impl::post_ring(const render::GradeChoice& choice, std::int64_t from_pts,
                                    std::uint32_t epoch) {
  if (!grade_ring || !ring_live) return;
  const pl_color_space target = target_color;
  ring_serial = grade_serial;
  ring_target = target;
  GradeVersion version{.epoch = epoch, .from_pts_us = from_pts == kNoPts ? 0 : from_pts};
  // Le grade en trois passes vers une sortie PQ seulement : les autres corps et le SDR restent à l'écran.
  std::optional<render::GradeParams> params;
  if (grade && grade_body == render::GradeBody::Fused && target.transfer == PL_COLOR_TRC_PQ)
    params = grade_for(choice, target);
  const auto post_off = [&] {
    if (ring_posted && !ring_on) return;
    version.off = true;
    ring_live->post_grade(std::move(version));
    ring_posted = true;
    ring_on = false;
  };
  if (!params) {
    post_off();
    return;
  }
  // L'entrée SDR → PQ de libplacebo pour ce couple source, cible, relevée par une rampe sans hook.
  if (!ring_nits || !pl_color_space_equal(&ring_input_source, &source) ||
      !pl_color_space_equal(&ring_input_target, &target)) {
    if (ramp_renderer == nullptr) ramp_renderer = pl_renderer_create(log, vk->gpu);
    auto captured = render::capture_grade_input(ramp_renderer, vk->gpu, source, target);
    if (!captured) {
      channel->warn("grade de l'anneau : entrée non relevée ({}), l'écran garde le grade",
                    captured.error().detail);
      ring_nits.reset();
      post_off();
      return;
    }
    std::vector<float> nits;
    nits.reserve(3 * static_cast<std::size_t>(render::kGradeInputPoints));
    for (const std::vector<float>& channel_nits : captured->nits)
      nits.insert(nits.end(), channel_nits.begin(), channel_nits.end());
    ring_nits = std::make_shared<const std::vector<float>>(std::move(nits));
    ring_matrix = captured->matrix;
    ring_input_source = source;
    ring_input_target = target;
  }
  // KWin comme match_kwin le poserait pour ce pic.
  std::optional<render::Grade::KwinMatch> kwin;
  if (kde_sdr_nits) {
    const render::KwinHint hint = render::kwin_hint(params->peak, *kde_sdr_nits);
    kwin = render::Grade::KwinMatch{.reference = *kde_sdr_nits, .peak = hint.panel_peak_nits};
  }
  for (const auto& field : kernels::kGradeUniformFields) {
    const std::optional<float> value = grade->param_for(field.name, *params, kwin);
    if (!value) {
      channel->warn("grade de l'anneau : paramètre {} absent du grade, l'écran garde le grade", field.name);
      post_off();
      return;
    }
    version.uniforms.*field.field = *value;
  }
  version.uniforms.skin_on = grade->skin_mask() ? 1.0F : 0.0F;
  if (std::optional<render::CubeLut> stack = grade->look_stack_for(*params)) {
    if (!ring_look || *ring_look != stack->rgb)
      ring_look = std::make_shared<const std::vector<float>>(std::move(stack->rgb));
    version.look_size = stack->size;
    version.look_rgb = ring_look;
  }
  version.input_nits = ring_nits;
  version.input_matrix = ring_matrix;
  version.peak_nits = target.hdr.max_luma;
  const std::uint32_t posted = ring_live->post_grade(std::move(version));
  ring_versions[ring_next++ % ring_versions.size()] = RingVersion{.version = posted, .space = target};
  ring_posted = true;
  ring_on = true;
}

void PlaceboScreen::Impl::follow_ring() {
  if (!grade_ring || !ring_live) return;
  if (std::exchange(ring_absorb, false)) {  // le choix posté d'avance vient de passer à l'écran
    ring_serial = grade_serial;
    ring_target = target_color;
    return;
  }
  if (ring_posted && ring_serial == grade_serial && pl_color_space_equal(&ring_target, &target_color)) return;
  post_ring(grade_choice, shown_pts == kNoPts ? 0 : shown_pts + 1, shown_epoch);
}

}  // namespace ajp::app::play
