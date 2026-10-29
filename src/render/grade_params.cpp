#include "render/grade_params.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <string_view>

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

// Émulation TrueHDR (lois ajustées sur le SDK RTX Video, hdr-grade.lua:113-127) : contraste en
// puissance autour du pivot, saturation sur la chroma, gris moyen m placé vers m nits.
void apply_truehdr_look(const GradeSettings& s, double peak, GradeParams& p) {
  const double contrast = (std::min(s.contrast, grade_limits::kSdkContrastMax) - 100.0) / 100.0;
  const double saturation = (std::min(s.saturation, grade_limits::kSdkContrastMax) - 100.0) / 100.0;
  const double middle = s.middle_gray;
  p.emulate = 1.0;
  p.em_base = std::clamp(4.0 * middle, 40.0, std::min(400.0, 0.5 * peak));
  p.em_pow = std::clamp(6.0 * std::pow(50.0 / middle, 0.35), 2.0, 16.0);
  p.em_lo = 1.0 + (contrast >= 0.0 ? 0.38 : 0.35) * contrast;
  p.em_hi = 1.0 + (contrast >= 0.0 ? 0.28 : 0.26) * contrast;
  p.em_sat = 1.0 + 0.19 * saturation;
}

// La fin du grade : le contraste et la saturation que RTX HDR ne donne plus (au-delà de 200 sur les curseurs)
// plus ceux du préréglage. Au repos (curseurs à 200 ou moins, préréglage sans punch), identité exacte.
void apply_punch(const GradeSettings& s, GradeParams& p) {
  p.punch = s.punch_base + std::max(0.0, s.contrast - grade_limits::kSdkContrastMax) / 200.0;
  p.chroma = 1.0 + s.chroma_base + 1.2 * std::max(0.0, s.saturation - grade_limits::kSdkContrastMax) / 200.0;
  p.pivot = 0.4 * grade_limits::kSdrWhiteNits;
}

// Sortie SDR : le blanc de la source est celui de l'écran, aucune marge pour des éclats. Restent les
// noirs, les couleurs (le curseur de RTX HDR), le garde-fou, le vrai noir, et de l'aspect TrueHDR
// son contraste et sa saturation autour du gris moyen.
GradeParams sdr_params(const GradeSettings& s) {
  GradeParams p;
  p.sdr_out = 1.0;
  p.peak = grade_limits::kSdrWhiteNits;
  p.paper = p.peak;
  p.in_max = p.peak;
  p.shadows = s.shadows;
  p.shadow_knee = 0.25 * p.peak;
  p.area_gain = 1.0;
  p.spec_gain = 1.0;
  p.vibrance = s.vibrance_rtx;
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
  return p;
}

}  // namespace

GradeSettings grade_preset(GradePreset preset, GradeSettings custom) {
  switch (preset) {
    case GradePreset::Naturel:
      return GradeSettings{};
    case GradePreset::Eclatant: {
      GradeSettings s;
      s.shadows = 0.50;
      s.highlights = 0.95;
      s.highlights_rtx = 0.90;
      s.vibrance = 0.60;
      s.vibrance_rtx = 0.60;
      s.area_rtx = 0.50;
      s.guard = 0.80;
      s.expand = 0.70;
      s.punch_base = 0.30;   // des noirs plus profonds, des hautes lumières qui claquent
      s.chroma_base = 0.12;  // des couleurs plus riches, sans toucher aux gris
      return s;
    }
    case GradePreset::Custom:
      return custom;
  }
  return custom;
}

std::string_view preset_key(GradePreset preset) {
  switch (preset) {
    case GradePreset::Eclatant:
      return "eclatant";
    case GradePreset::Naturel:
      return "naturel";
    case GradePreset::Custom:
      return "perso";
  }
  return "perso";
}

std::string_view preset_label(GradePreset preset) {
  switch (preset) {
    case GradePreset::Eclatant:
      return "Vivid";
    case GradePreset::Naturel:
      return "Natural";
    case GradePreset::Custom:
      return "Custom";
  }
  return "Custom";
}

std::expected<void, Refusal> check_grade_settings(const GradeSettings& s) {
  using namespace grade_limits;
  for (const auto& result :
       {check("paper_white", s.paper_white, kPaperWhite), check("shadows", s.shadows, kShadows),
        check("highlights", s.highlights, kHighlights),
        check("highlights_rtx", s.highlights_rtx, kHighlights), check("vibrance", s.vibrance, kVibrance),
        check("vibrance_rtx", s.vibrance_rtx, kVibrance), check("contrast", s.contrast, kContrast),
        check("saturation", s.saturation, kSaturation), check("middle_gray", s.middle_gray, kMiddleGray),
        check("subtitle_white", s.subtitle_white, kSubtitleWhite), check("black_lo", s.black_lo, kBlackNits),
        check("black_hi", s.black_hi, kBlackNits), check("guard", s.guard, kGuard),
        check("area_rtx", s.area_rtx, kAreaRtx), check("expand", s.expand, kExpand),
        check("abl_lo", s.abl_lo, kAblNits), check("abl_hi", s.abl_hi, kAblNits)}) {
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
  } else if (context.route == core::HdrMode::TrueHdr) {
    decision.mode = GradeMode::TrueHdr;
  }
  if (decision.mode == GradeMode::Neutral) return decision;

  GradeParams& p = decision.params;
  p.peak = peak;
  p.paper = paper;
  p.shadows = s.shadows;
  p.shadow_knee = 0.25 * paper;
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
    p.area_gain = std::min(1.0 + 0.5 * s.highlights, room);
    p.spec_gain = std::max(p.area_gain, 1.0 + s.highlights * (room - 1.0));
    p.vibrance = s.vibrance;
  } else {
    // TrueHDR, son émulation et le HDR natif atteignent déjà le pic : les aplats restent, les
    // éclats isolés montent, à partir de la moitié du pic.
    p.paper = 0.5 * peak;
    p.in_max = peak;
    p.area_gain = 1.0 + 0.8 * s.area_rtx;
    // 1 + 1,2 h comme le lecteur actuel jusqu'à h = 0,25 (son réglage par défaut), puis plus raide : à h = 1
    // les éclats montent à 3,4 fois leur niveau au lieu de 2,2.
    p.spec_gain =
        std::max(p.area_gain, 1.0 + 1.2 * s.highlights_rtx + 1.6 * std::max(0.0, s.highlights_rtx - 0.25));
    p.vibrance = s.vibrance_rtx;
  }
  // Dans les plages //!PARAM du hook.
  p.area_gain = std::clamp(p.area_gain, 1.0, 4.0);
  p.spec_gain = std::clamp(p.spec_gain, p.area_gain, 20.0);
  // Sans vrai noir (écran non OLED, ou coupé), la dixième passe est sautée.
  p.black_lo = s.true_black ? s.black_lo : 0.0;
  p.black_hi = s.true_black ? s.black_hi : 0.0;
  apply_punch(s, p);
  return decision;
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
