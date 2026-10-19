#include "render/grade_params.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <string_view>

namespace ajp::render {
namespace {

std::expected<void, Refusal> check(std::string_view field, double value, core::Range<double> range) {
  if (std::isnan(value) || !range.contains(value)) {
    return std::unexpected(Refusal{RefusalCode::FieldOutOfRange,
                                   std::format("grade.{} = {} ({} à {})", field, value, range.lo, range.hi)});
  }
  return {};
}

// Émulation TrueHDR (lois ajustées sur le SDK RTX Video, hdr-grade.lua:113-127) : contraste en
// puissance autour du pivot, saturation sur la chroma, gris moyen m placé vers m nits.
void apply_truehdr_look(const GradeSettings& s, double peak, GradeParams& p) {
  const double contrast = (s.contrast - 100.0) / 100.0;
  const double saturation = (s.saturation - 100.0) / 100.0;
  const double middle = s.middle_gray;
  p.emulate = 1.0;
  p.em_base = std::clamp(4.0 * middle, 40.0, std::min(400.0, 0.5 * peak));
  p.em_pow = std::clamp(6.0 * std::pow(50.0 / middle, 0.35), 2.0, 16.0);
  p.em_lo = 1.0 + (contrast >= 0.0 ? 0.38 : 0.35) * contrast;
  p.em_hi = 1.0 + (contrast >= 0.0 ? 0.28 : 0.26) * contrast;
  p.em_sat = 1.0 + 0.19 * saturation;
}

}  // namespace

GradeSettings grade_preset(GradePreset preset, GradeSettings custom) {
  switch (preset) {
    case GradePreset::Naturel:
      return GradeSettings{};
    case GradePreset::Eclatant: {
      GradeSettings s;
      s.shadows = 0.40;
      s.highlights = 0.90;
      s.highlights_rtx = 0.80;
      s.vibrance = 0.50;
      s.vibrance_rtx = 0.45;
      s.area_rtx = 0.35;
      s.guard = 0.80;
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
      return "Éclatant";
    case GradePreset::Naturel:
      return "Naturel";
    case GradePreset::Custom:
      return "Personnalisé";
  }
  return "Personnalisé";
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
        check("area_rtx", s.area_rtx, kAreaRtx)}) {
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
  GradeDecision decision;
  decision.source_white_nits = native_sdr && s.truehdr_look ? grade_limits::kSdrWhiteNits : paper;

  if (!s.grade || !context.enabled || !context.output_pq) return decision;
  if (context.source_hdr) {
    decision.mode = s.grade_native_hdr ? GradeMode::NativeHdr : GradeMode::Neutral;
  } else if (native_sdr) {
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
  p.guard = s.guard;
  p.emulate = 0.0;
  p.sdr_white = grade_limits::kSdrWhiteNits;
  const bool look = decision.mode == GradeMode::Native && s.truehdr_look;
  if (look) apply_truehdr_look(s, peak, p);
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
    p.area_gain = 1.0 + 0.5 * s.area_rtx;
    p.spec_gain = std::max(p.area_gain, 1.0 + 1.2 * s.highlights_rtx);
    p.vibrance = s.vibrance_rtx;
  }
  // Dans les plages //!PARAM du hook.
  p.area_gain = std::clamp(p.area_gain, 1.0, 4.0);
  p.spec_gain = std::clamp(p.spec_gain, p.area_gain, 20.0);
  // Le vrai noir n'a de sens que sur OLED ; ailleurs la dixième passe est sautée.
  p.black_lo = s.oled ? s.black_lo : 0.0;
  p.black_hi = s.oled ? s.black_hi : 0.0;
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

}  // namespace ajp::render
