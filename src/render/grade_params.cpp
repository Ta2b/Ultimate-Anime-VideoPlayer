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
double vibrance_effect(double vibrance) {
  return response(vibrance, 0.8, 1.6);
}
// Le pied des noirs s'élargit avec le curseur : plus de gris moyens-sombres sont creusés, pas seulement
// l'extrême bas.
double shadow_knee_for(double shadows, double white) {
  return white * (0.25 + 0.30 * std::min(shadows / 0.8, 1.0));
}

// Émulation TrueHDR (lois ajustées sur le SDK RTX Video, hdr-grade.lua:113-127) : contraste en
// puissance autour du pivot, saturation sur la chroma, gris moyen m placé vers m nits.
void apply_truehdr_look(const GradeSettings& s, double peak, GradeParams& p) {
  const double contrast = (std::min(s.contrast, grade_limits::kSdkContrastMax) - 100.0) / 100.0;
  const double middle = s.middle_gray;
  p.emulate = 1.0;
  p.em_base = std::clamp(4.0 * middle, 40.0, std::min(400.0, 0.5 * peak));
  p.em_pow = std::clamp(6.0 * std::pow(50.0 / middle, 0.35), 2.0, 16.0);
  p.em_lo = 1.0 + (contrast >= 0.0 ? 0.38 : 0.35) * contrast;
  p.em_hi = 1.0 + (contrast >= 0.0 ? 0.28 : 0.26) * contrast;
  p.em_sat = grade_limits::truehdr_saturation_factor(grade_limits::effective_saturation(s.saturation));
}

// Code PQ (SMPTE ST 2084) d'une luminance en nits.
double pq_code(double nits) {
  const double y = std::clamp(nits / 10000.0, 0.0, 1.0);
  const double q = std::pow(y, 0.1593017578125);
  return std::pow((0.8359375 + 18.8515625 * q) / (1.0 + 18.6875 * q), 78.84375);
}

// Grade adaptatif (HDR adaptatif, sortie SDR gradée : tous les GPU) : le shader
// glisse noirs, vibrance et courbe TrueHDR émulée (gris moyen) de « scène sombre » (les curseurs) vers «
// scène claire » selon le code PQ moyen très lissé de la scène. Les seuils (luminance en % de la source,
// gamma 2,4, blanc de référence 203 nits) sont convertis en codes PQ. Sans émulation, la courbe ne bouge pas
// (les valeurs claires reprennent les sombres).
// La scène du lecteur, à l'échelle de l'image du grade : le blanc SDR y vaut sdr_white.
void apply_scene_light(const GradeSettings& s, GradeParams& p) {
  if (!s.scene_light) return;
  p.fs_apl = std::max(s.scene_light->apl_nits, 0.0) * p.sdr_white / grade_limits::kSdrWhiteNits;
  p.fs_bright = std::clamp(s.scene_light->lit, 0.0, 1.0);
  p.an_lift = kAnimeLift * std::exp2(std::clamp(s.anime_exposure, -kCurveMaxE, kCurveMaxE));
}

void apply_adaptive(const GradeSettings& s, double peak, bool sdr_out, GradeParams& p) {
  if (!s.adaptive) return;
  // Le code que la passe des statistiques compare : le PQ de la luminance de la source, sur toutes les
  // sorties (en sortie SDR aussi, la mesure est en PQ).
  const auto luma_to_pq = [](double percent) {
    return pq_code(grade_limits::kSdrWhiteNits * std::pow(percent / 100.0, 2.4));
  };
  p.ad_on = 1.0;
  p.ad_lo = luma_to_pq(s.adaptive_lo);
  p.ad_hi = luma_to_pq(s.adaptive_hi);
  if (s.adaptive_force >= 0.0) {  // les seuils hors de [0, 1] : le smoothstep du shader rend 0 ou 1
    const bool bright = grade_limits::forced_bright(s.adaptive_force);
    p.ad_lo = bright ? -2.0 : 2.0;
    p.ad_hi = bright ? -1.0 : 3.0;
    p.gl_force = grade_limits::forced_glow(s.adaptive_force) ? 1.0 : 0.0;
  } else if (s.adaptive_shot) {
    p.sh_on = 1.0;
    p.sh_t = std::clamp(s.adaptive_shot->t, 0.0, 1.0);
    p.sh_glow = std::clamp(s.adaptive_shot->glow, 0.0, 1.0);
    p.sh_sky = std::clamp(s.adaptive_shot->tv, 0.0, 1.0);
    p.sh_series = std::clamp(s.adaptive_shot->series, 0.0, 1.0);
  }
  p.ad_shadows = shadows_effect(s.adaptive_shadows);
  // La vibrance suit la couleur (ciel bleu saturé), que le shader ne mesure pas : seule la part retirée
  // dans les scènes claires l'adapte sur ces routes.
  p.ad_vibrance = p.vibrance * (1.0 - std::clamp(s.adaptive_vibrance_cut, 0.0, 1.0));
  p.ad_sky_vibrance = vibrance_effect(s.adaptive_vibrance);
  p.ad_em_base = p.em_base;
  p.ad_em_pow = p.em_pow;
  // La scène sombre avec éclat : ses noirs et ses couleurs à elle, sa courbe si elle est émulée.
  p.gl_shadows = shadows_effect(s.adaptive_glow_shadows);
  p.gl_vibrance = vibrance_effect(s.adaptive_glow_vibrance);
  p.gl_em_base = p.em_base;
  p.gl_em_pow = p.em_pow;
  // Sortie SDR : le blanc est fixe, un gris plus bas veut dire « plus sombre » et non « moins d'expansion » :
  // l'adaptatif n'y règle que les noirs et les couleurs, la courbe reste celle du gris de base.
  if (p.emulate > 0.5 && !sdr_out) {
    const double top = std::min(400.0, 0.5 * peak);
    const auto base = [top](double gray) { return std::clamp(4.0 * gray, 40.0, top); };
    const auto power = [](double gray) { return std::clamp(6.0 * std::pow(50.0 / gray, 0.35), 2.0, 16.0); };
    p.ad_em_base = base(s.adaptive_gray);
    p.ad_em_pow = power(s.adaptive_gray);
    p.gl_em_base = base(s.adaptive_glow_gray);
    p.gl_em_pow = power(s.adaptive_glow_gray);
  }
}

// La fin du grade : le contraste et la saturation du préréglage. Sans punch ni chroma de base, identité
// exacte (les curseurs de contraste et de saturation, 0 à 200, sont ceux de l'aspect TrueHDR émulé).
void apply_punch(const GradeSettings& s, GradeParams& p) {
  p.punch = s.punch_base;
  p.chroma = 1.0 + s.chroma_base;
  p.pivot = 0.4 * grade_limits::kSdrWhiteNits;
  // Le peps de la méthode « HDR anime » : 100 % partout, la méthode telle quelle.
  p.an_punch = std::clamp((s.anime_contrast - 100) / 100.0, -1.0, 2.0);
  p.an_chroma = std::clamp(s.anime_saturation / 100.0, 0.0, 2.0);
  p.an_spec = std::clamp(s.anime_highlights / 100.0, 0.0, 2.0);
}

// Sortie SDR : le blanc de la source est celui de l'écran, aucune marge pour des éclats. Restent les
// noirs, les couleurs, le garde-fou, le vrai noir, et de l'aspect TrueHDR
// son contraste et sa saturation autour du gris moyen.
GradeParams sdr_params(const GradeSettings& s) {
  GradeParams p;
  p.sdr_out = 1.0;
  p.peak = grade_limits::kSdrWhiteNits;
  p.paper = p.peak;
  p.in_max = p.peak;
  p.shadows = shadows_effect(s.shadows);
  p.shadow_knee = shadow_knee_for(s.shadows, p.peak);
  p.area_gain = 1.0;
  p.spec_gain = 1.0;
  p.vibrance = vibrance_effect(s.vibrance_rtx);
  p.abl = 0.0;
  p.guard = s.guard;
  p.expand = 0.0;
  p.gamut_p3 = 0.0;
  p.sdr_white = p.peak;
  p.emulate = 0.0;
  if (s.truehdr_look) {
    apply_truehdr_look(s, p.peak, p);
    // L'aspect TrueHDR entier (emulate = 1) dans le blanc SDR : la courbe envoie toujours le blanc
    // à peak et relève les sombres comme TrueHDR (s2 = idx^(2/2,4)) avant le contraste sous le
    // pivot. Le contraste seul, sans ce relèvement, écrasait les noirs : à contraste 200, la
    // médiane d'une scène de nuit passait de 2,9 à 0,46 nit (−84 %, contre −31 % en HDR10).
    p.em_base = std::clamp(4.0 * s.middle_gray, 40.0, 400.0);  // le pivot : 0,2575 × em_base
  }
  p.black_lo = s.true_black ? s.black_lo : 0.0;
  p.black_hi = s.true_black ? s.black_hi : 0.0;
  apply_punch(s, p);
  apply_adaptive(s, p.peak, true, p);
  apply_scene_light(s, p);
  return p;
}

}  // namespace

std::expected<void, Refusal> check_grade_settings(const GradeSettings& s) {
  using namespace grade_limits;
  for (const auto& result : {check("paper_white", s.paper_white, kPaperWhite),
                             check("shadows", s.shadows, kShadows),
                             check("highlights", s.highlights, kHighlights),
                             check("highlights_rtx", s.highlights_rtx, kHighlights),
                             check("vibrance", s.vibrance, kVibrance),
                             check("vibrance_rtx", s.vibrance_rtx, kVibrance),
                             check("contrast", s.contrast, kContrast),
                             check("saturation", s.saturation, kSaturation),
                             check("middle_gray", s.middle_gray, kMiddleGray),
                             check("subtitle_white", s.subtitle_white, kSubtitleWhite),
                             check("black_lo", s.black_lo, kBlackNits),
                             check("black_hi", s.black_hi, kBlackNits),
                             check("guard", s.guard, kGuard),
                             check("area_rtx", s.area_rtx, kAreaRtx),
                             check("expand", s.expand, kExpand),
                             check("abl_lo", s.abl_lo, kAblNits),
                             check("abl_hi", s.abl_hi, kAblNits),
                             check("adaptive_lo", s.adaptive_lo, kAdaptiveLuma),
                             check("adaptive_hi", s.adaptive_hi, kAdaptiveLuma),
                             check("adaptive_shadows", s.adaptive_shadows, kShadows),
                             check("adaptive_vibrance", s.adaptive_vibrance, kVibrance),
                             check("adaptive_vibrance_cut", s.adaptive_vibrance_cut, kAdaptiveCut),
                             check("adaptive_gray", s.adaptive_gray, kMiddleGray),
                             check("adaptive_glow_gray", s.adaptive_glow_gray, kMiddleGray),
                             check("adaptive_glow_shadows", s.adaptive_glow_shadows, kShadows),
                             check("adaptive_glow_vibrance", s.adaptive_glow_vibrance, kVibrance)}) {
    if (!result) return result;
  }
  if (!(s.abl_hi > s.abl_lo)) {
    return std::unexpected(
        Refusal{RefusalCode::FieldOutOfRange,
                std::format("grade.abl_hi = {} (au-dessus de abl_lo = {})", s.abl_hi, s.abl_lo)});
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
  // Le mode SDR sur une sortie HDR10 : l'image SDR est gradée dans le conteneur PQ comme la route native sans
  // l'aspect TrueHDR (blanc papier, éclats limités), au lieu de rester sans grade (retour de Tatoub).
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
  p.gamut_p3 = s.gamut_p3 ? 1.0 : 0.0;
  // OLED : les éclats sont modérés sur les scènes claires, où le panneau limite sa puissance.
  p.abl = s.oled ? 0.6 : 0.0;
  p.abl_lo = s.abl_lo;
  p.abl_hi = s.abl_hi;
  p.guard = s.guard;
  p.expand = s.expand;
  p.emulate = 0.0;
  p.sdr_white = grade_limits::kSdrWhiteNits;
  const bool look =
      native_sdr && s.truehdr_look;  // l'aspect adaptatif : la route native seule, pas le mode SDR
  if (look) {
    apply_truehdr_look(s, peak, p);
    p.adapt = 1.0;  // le HDR adaptatif : la route native de tous les GPU suit la scène
  }
  if (decision.mode == GradeMode::Native && !look) {
    // Le SDR s'arrête au blanc papier : un blanc isolé peut atteindre le pic, un grand aplat
    // blanc seulement +50 % à highlights = 1.
    const double room = peak / paper;
    p.in_max = paper;
    // Les curseurs visibles (Éclats, Aplats, Couleurs : ceux de l'interface sur une sortie HDR) règlent aussi
    // ce mode ; leurs défauts (0,25 ; 0 ; 0,10) donnent les valeurs de départ d'avant (0,50 ; 0,15), pour que
    // le rendu par défaut ne change pas.
    const double h = response(std::min(2.0 * s.highlights_rtx, 1.0), 1.0, 1.0);
    const double areas = response(s.area_rtx, 1.0, 1.0);
    p.area_gain = std::min(1.0 + 0.8 * (h + areas), room);
    p.spec_gain = std::max(p.area_gain, 1.0 + h * (room - 1.0));
    p.vibrance = vibrance_effect(std::min(1.5 * s.vibrance_rtx, 1.0));
  } else {
    // L'aspect TrueHDR émulé et le HDR natif atteignent déjà le pic : les aplats restent, les
    // éclats isolés montent, à partir de la moitié du pic.
    p.paper = 0.5 * peak;
    p.in_max = peak;
    const double area = response(s.area_rtx, 1.0, 1.0);
    const double high = response(s.highlights_rtx, 1.0, 1.0);
    p.area_gain = 1.0 + 0.8 * area;
    // 1 + 1,2 h comme le lecteur actuel jusqu'à h = 0,25 (son réglage par défaut), puis plus raide : à h = 1
    // les éclats montent à 3,4 fois leur niveau au lieu de 2,2.
    p.spec_gain = std::max(p.area_gain, 1.0 + 1.2 * high + 1.6 * std::max(0.0, high - 0.25));
    p.vibrance = vibrance_effect(s.vibrance_rtx);
  }
  // Dans les plages //!PARAM du hook.
  p.area_gain = std::clamp(p.area_gain, 1.0, 4.0);
  p.spec_gain = std::clamp(p.spec_gain, p.area_gain, 20.0);
  // Sans vrai noir (écran non OLED, ou coupé), la dixième passe est sautée.
  p.black_lo = s.true_black ? s.black_lo : 0.0;
  p.black_hi = s.true_black ? s.black_hi : 0.0;
  apply_punch(s, p);
  apply_adaptive(s, peak, false, p);
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

std::optional<AblRange> abl_range(double frame_average_nits) {
  const AblRange range{.lo = 0.30 * frame_average_nits, .hi = 0.94 * frame_average_nits};
  if (std::isnan(frame_average_nits) || !grade_limits::kAblNits.contains(range.lo) ||
      !grade_limits::kAblNits.contains(range.hi))
    return std::nullopt;
  return range;
}

}  // namespace ajp::render
