#include "render/grade_params.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <string_view>

#include "render/curve.h"

namespace ajp::render {
namespace {

// Un curseur float relu en double dépasse sa borne d'un cheveu (0,8f vaut 0,800000012) : la
// marge de 1e-6 garde les play.toml écrits ainsi, sans ouvrir la plage.
std::expected<void, Refusal> check(std::string_view field, double value, core::Range<double> range) {
  const double slack = 1e-6 * std::max(1.0, std::abs(range.hi));
  if (std::isnan(value) || value < range.lo - slack || value > range.hi + slack) {
    return std::unexpected(Refusal{RefusalCode::FieldOutOfRange,
                                   std::format("grade.{} = {} ({} à {})", field, value, range.lo, range.hi)});
  }
  return {};
}

// Réponse des curseurs du grade : la course (0 à `hi`) ne change pas, l'effet si. Une puissance < 1 relève
// les positions basses et moyennes (le curseur « prend » dès qu'on le touche), `gain` > 1 pousse la butée.
double response(double x, double hi, double gain, double exponent = 0.7) {
  if (!(x > 0.0)) return 0.0;
  return gain * hi * std::pow(std::min(x, hi) / hi, exponent);
}
double shadows_effect(double shadows) {
  return std::min(response(shadows, 0.8, 1.2), 0.95);
}
// Le pied des noirs s'élargit avec le curseur : plus de gris moyens-sombres sont creusés, pas seulement
// l'extrême bas.
double shadow_knee_for(double shadows, double white) {
  return white * (0.25 + 0.30 * std::min(shadows / 0.8, 1.0));
}

// La scène du lecteur, à l'échelle de l'image du grade : le blanc SDR y vaut sdr_white.
void apply_scene_light(const GradeSettings& s, GradeParams& p) {
  if (s.series_apl_nits > 0.0) p.fs_series = s.series_apl_nits * p.sdr_white / grade_limits::kSdrWhiteNits;
  if (!s.scene_light) return;
  p.fs_apl = std::max(s.scene_light->apl_nits, 0.0) * p.sdr_white / grade_limits::kSdrWhiteNits;
  p.an_lift = kAnimeLift * std::exp2(std::clamp(s.anime_exposure, -kCurveMaxE, kCurveMaxE));
}

// La saturation du préréglage, le pivot et le peps de la passe anime (100 % : la méthode telle quelle).
void apply_anime(const GradeSettings& s, GradeParams& p) {
  p.chroma = 1.0 + s.chroma_base;
  p.pivot = 0.4 * grade_limits::kSdrWhiteNits;
  p.an_punch = std::clamp((s.anime_contrast - 100) / 100.0, -1.0, 2.0);
  p.an_chroma = std::clamp(s.anime_saturation / 100.0, 0.0, 2.0);
  p.an_spec = std::clamp(s.anime_highlights / 100.0, 0.0, 2.0);
}

// Sortie SDR : le blanc de la source est celui de l'écran, aucune marge pour des éclats. Restent les
// noirs, le vrai noir et le peps de la passe anime.
GradeParams sdr_params(const GradeSettings& s) {
  GradeParams p;
  p.sdr_out = 1.0;
  p.peak = grade_limits::kSdrWhiteNits;
  p.paper = p.peak;
  p.shadows = shadows_effect(s.shadows);
  p.shadow_knee = shadow_knee_for(s.shadows, p.peak);
  p.spec_gain = 1.0;
  p.sdr_white = p.peak;
  p.black_lo = s.true_black ? s.black_lo : 0.0;
  p.black_hi = s.true_black ? s.black_hi : 0.0;
  apply_anime(s, p);
  apply_scene_light(s, p);
  return p;
}

}  // namespace

std::expected<void, Refusal> check_grade_settings(const GradeSettings& s) {
  using namespace grade_limits;
  for (const auto& result :
       {check("paper_white", s.paper_white, kPaperWhite), check("shadows", s.shadows, kShadows),
        check("highlights_rtx", s.highlights_rtx, kHighlights), check("black_lo", s.black_lo, kBlackNits),
        check("black_hi", s.black_hi, kBlackNits), check("area_rtx", s.area_rtx, kAreaRtx)}) {
    if (!result) return result;
  }
  return {};
}

std::expected<double, Refusal> grade_peak(const core::DisplayInfo& display) {
  if (display.provenance.peak == core::Provenance::Default) return grade_limits::kDefaultPeakNits;
  if (auto ok = check("peak", display.peak_nits, grade_limits::kPeakNits); !ok)
    return std::unexpected(ok.error());
  return display.peak_nits;
}

std::expected<GradeDecision, Refusal> decide_grade(const GradeSettings& s, const GradeContext& context) {
  if (auto ok = check_grade_settings(s); !ok) return std::unexpected(ok.error());
  if (auto ok = check("peak", context.peak_nits, grade_limits::kPeakNits); !ok)
    return std::unexpected(ok.error());

  const double peak = context.peak_nits;
  // Le blanc papier reste sous le pic, pour laisser de la marge aux éclats.
  const double paper = std::floor(std::min(s.paper_white, 0.75 * peak) + 0.5);
  const bool native_sdr = context.route == core::HdrMode::Native && !context.source_hdr;
  // Le mode SDR sur une sortie HDR10 : l'image SDR est gradée dans le conteneur PQ comme la route native au
  // blanc papier (éclats limités), au lieu de rester sans grade (retour de Tatoub).
  const bool plain_sdr = context.route == core::HdrMode::Off && !context.source_hdr;
  GradeDecision decision;
  decision.source_white_nits = native_sdr && s.truehdr_look ? grade_limits::kSdrWhiteNits : paper;

  if (!s.grade || !context.enabled) return decision;
  if (!context.output_pq) {
    if (context.source_hdr) return decision;
    decision.mode = GradeMode::Sdr;
    decision.params = sdr_params(s);
    decision.source_white_nits = grade_limits::kSdrWhiteNits;
    return decision;
  }
  if (context.source_hdr) {
    decision.mode = s.grade_native_hdr ? GradeMode::NativeHdr : GradeMode::Neutral;
  } else if (native_sdr || plain_sdr) {
    decision.mode = GradeMode::Native;
  }
  if (decision.mode == GradeMode::Neutral) return decision;

  GradeParams& p = decision.params;
  p.peak = peak;
  p.paper = paper;
  p.shadows = shadows_effect(s.shadows);
  p.shadow_knee = shadow_knee_for(s.shadows, paper);
  p.sdr_white = grade_limits::kSdrWhiteNits;
  // La route native (le défaut) place le blanc de la source à 203 nits et les aplats à la moitié du pic ; le
  // mode SDR sur une sortie HDR10 et une source HDR gradée gardent le blanc papier.
  const bool native = native_sdr && s.truehdr_look;
  double area = 1.0;
  if (native) p.adapt = 1.0;
  if (decision.mode == GradeMode::Native && !native) {
    // Le SDR s'arrête au blanc papier : un blanc isolé peut atteindre le pic, un grand aplat blanc
    // seulement +50 % à highlights = 1.
    const double room = peak / paper;
    const double h = response(std::min(2.0 * s.highlights_rtx, 1.0), 1.0, 1.0);
    area = std::min(1.0 + 0.8 * (h + response(s.area_rtx, 1.0, 1.0)), room);
    p.spec_gain = std::max(area, 1.0 + h * (room - 1.0));
  } else {
    // Les éclats isolés montent à partir de la moitié du pic : 1 + 1,2 h jusqu'à h = 0,25, puis plus raide (à
    // h = 1, 3,4 fois leur niveau).
    p.paper = 0.5 * peak;
    const double high = response(s.highlights_rtx, 1.0, 1.0);
    area = 1.0 + 0.8 * response(s.area_rtx, 1.0, 1.0);
    p.spec_gain = std::max(area, 1.0 + 1.2 * high + 1.6 * std::max(0.0, high - 0.25));
  }
  area = std::clamp(area, 1.0, 4.0);
  p.spec_gain = std::clamp(p.spec_gain, area, 20.0);
  // Sans vrai noir (écran non OLED, ou coupé), la passe est sautée.
  p.black_lo = s.true_black ? s.black_lo : 0.0;
  p.black_hi = s.true_black ? s.black_hi : 0.0;
  apply_anime(s, p);
  apply_scene_light(s, p);
  return decision;
}

void sdr_tonemap(GradeParams& p) {
  // Sans la pile de LUT : réglée pour compenser la compression de KWin, elle porte le blanc SDR vers 5 500
  // nits, et la courbe vers le SDR en écrêtait le quart de la plage au blanc (audit F5).
  for (double GradeParams::* mix :
       {&GradeParams::look_mix, &GradeParams::look_mix2, &GradeParams::look_mix3, &GradeParams::look_mix4,
        &GradeParams::look_mix5, &GradeParams::look_mix6, &GradeParams::look_mix7, &GradeParams::look_mix8,
        &GradeParams::look_mix9, &GradeParams::look_mix10, &GradeParams::look_mix11, &GradeParams::look_mix12,
        &GradeParams::look_mix13, &GradeParams::look_mix14, &GradeParams::look_mix15})
    p.*mix = 0.0;
  p.look_pq = 0.0;
  p.sdr_tm = 1.0;
  p.sdr_out = 0.0;
  p.black_lo = 0.0;
  p.black_hi = 0.0;
  // Les mêmes bornes que 3-anime.hook : le blanc de la source, la clé de la scène, la marge des éclats.
  const double white = p.adapt > 0.5 ? p.sdr_white : p.paper;
  const double key = std::clamp((p.fs_apl >= 0.0 ? p.fs_apl : 0.15 * white) / p.sdr_white, 0.0, 1.0);
  const double t = std::clamp((key - 0.30) / 0.40, 0.0, 1.0);
  p.tm_white = p.sdr_white * p.an_lift;
  p.tm_top = std::max(p.peak * (1.0 - 0.5 * t * t * (3.0 - 2.0 * t)), 1.25 * p.tm_white);
}

std::optional<BlackThresholds> black_thresholds(std::span<const double> step_nits, double steps) {
  if (step_nits.size() < 2 || std::isnan(steps)) return std::nullopt;
  steps = std::clamp(steps, grade_limits::kBlackSteps.lo, grade_limits::kBlackSteps.hi);
  const double last = static_cast<double>(step_nits.size() - 1);
  const double at = std::min(steps, last);
  const auto below = static_cast<std::size_t>(std::floor(at));
  const std::size_t above = std::min(below + 1, step_nits.size() - 1);
  const double hi = std::lerp(step_nits[below], step_nits[above], at - static_cast<double>(below));
  const double lo = 0.5 * (step_nits[0] + step_nits[1]);
  if (!(hi > lo) || lo < 0.0) return std::nullopt;
  return BlackThresholds{.lo = lo, .hi = std::min(hi, grade_limits::kBlackNits.hi)};
}

}  // namespace ajp::render
