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
  impl_->source = color_space_of(color);
}

void PlaceboScreen::set_grade(const render::GradeChoice& choice) {
  impl_->grade_choice = choice;
  impl_->select_grade(impl_->target_color);
}

void PlaceboScreen::Impl::select_grade(const pl_color_space& target) {
  const std::optional<render::GradeParams> params = grade_for(grade_choice, target);
  grade_loaded_pq.reset();
  if (!grade || !params) return;
  if (*params != grade->params()) grade->set_params(*params);
  grade_loaded_pq = target.transfer == PL_COLOR_TRC_PQ;
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
    auto grade = render::Grade::create(gpu);
    if (!grade) return std::unexpected(std::move(grade.error()));
    s.ramp_grade.emplace(std::move(*grade));
  }
  render::GradeParams params = s.grade->params();
  params.black_lo = 0.0;  // le vrai noir se cale sur ce qui sort sans lui
  params.black_hi = 0.0;
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
  return nits;
}

}  // namespace ajp::app::play
