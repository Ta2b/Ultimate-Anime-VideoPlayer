// Réglages et paramètres du grade HDR, sans GPU.
//
// GradeSettings porte les réglages du profil (table des paramètres du grade dans le plan) ;
// decide_grade() en tire, pour une route, une source et un écran donnés, le mode du grade et
// les valeurs des //!PARAM des hooks. C'est la fonction M.plan de hdr-grade.lua du lecteur actuel
// (https://github.com/Ta2b/mpv-AnimeJaNai-HDR, commit 33ff79d,
// BuildMpvUpscale2xAnimeJaNai/mpv-upscale-2x_animejanai/portable_config/hdr-tools/hdr-grade.lua,
// lignes 58-160), reprise telle quelle ; tests/unit/render/grade_params_test.cpp fixe ses sorties.
#pragma once

#include <cstdint>
#include <expected>

#include "core/profile.h"
#include "core/resolve.h"
#include "core/types.h"

namespace ajp::render {

// Plages des réglages, incluses ; une valeur hors plage est refusée, jamais corrigée.
namespace grade_limits {
inline constexpr core::Range<double> kPaperWhite{80.0, 400.0};
inline constexpr core::Range<double> kShadows{0.0, 0.8};
inline constexpr core::Range<double> kHighlights{0.0, 1.0};
inline constexpr core::Range<double> kVibrance{0.0, 0.8};
inline constexpr core::Range<double> kContrast{0.0, 200.0};
inline constexpr core::Range<double> kSaturation{0.0, 200.0};
inline constexpr core::Range<double> kMiddleGray{10.0, 100.0};
inline constexpr core::Range<double> kSubtitleWhite{80.0, 1000.0};
inline constexpr core::Range<double> kBlackNits{0.0, 50.0};
inline constexpr core::Range<double> kPeakNits{100.0, 10000.0};
// Pic du grade quand l'écran ne le dit pas (DisplayInfo par défaut).
inline constexpr double kDefaultPeakNits = 1000.0;
// Entrée de l'émulation TrueHDR : le blanc SDR placé à 203 nits.
inline constexpr double kSdrWhiteNits = 203.0;
}  // namespace grade_limits

struct GradeSettings {
  bool grade = true;              // grade activé par le profil
  bool grade_native_hdr = false;  // grade aussi une source déjà HDR
  bool truehdr_look = true;       // route native : émule la courbe TrueHDR (mpv_truehdr_look)
  double paper_white = 203.0;     // nits, blanc papier
  double shadows = 0.15;
  double highlights = 0.50;      // route native sans émulation
  double highlights_rtx = 0.25;  // TrueHDR, son émulation, HDR natif
  double vibrance = 0.15;
  double vibrance_rtx = 0.10;
  double contrast = 100.0;  // TrueHDR et son émulation, comme rtx_contrast
  double saturation = 100.0;
  double middle_gray = 50.0;
  double subtitle_white = 200.0;  // nits, blanc des sous-titres en sortie HDR
  bool oled = false;              // écran OLED : limiteur des scènes claires (abl)
  bool gamut_p3 = false;          // limite au gamut Display P3 de l'écran
  double black_lo = 0.0;          // nits : vrai noir OLED sous ce seuil
  double black_hi = 0.0;          // nits : fin du pied ; <= black_lo, la passe est sautée
  friend bool operator==(const GradeSettings&, const GradeSettings&) = default;
};

// FieldOutOfRange en nommant le champ (« grade.paper_white »).
std::expected<void, Refusal> check_grade_settings(const GradeSettings& settings);

// Ce qui est gradé, et comment.
enum class GradeMode : std::uint8_t {
  Neutral,    // pas de grade : sortie non PQ, grade coupé, route Off ou source HDR non gradée
  Native,     // SDR converti par libplacebo en PQ (route native), avec ou sans émulation TrueHDR
  TrueHdr,    // SDR passé par TrueHDR : l'image arrive déjà en PQ au pic de l'écran
  NativeHdr,  // source déjà HDR, gradée parce que grade_native_hdr le demande
};

struct GradeContext {
  core::HdrMode route = core::HdrMode::Off;  // celle du Plan
  bool source_hdr = false;                   // source PQ ou HLG
  bool output_pq = false;                    // la cible est encodée en PQ
  bool enabled = true;                       // bascule A/B de l'utilisateur
  double peak_nits = grade_limits::kDefaultPeakNits;
};

// Les //!PARAM de anime-hdr-grade.hook et de anime-hdr-black.hook.
struct GradeParams {
  double peak = 1000.0;
  double paper = 203.0;
  double in_max = 203.0;
  double shadows = 0.30;
  double shadow_knee = 50.0;
  double area_gain = 1.4;
  double spec_gain = 4.3;
  double vibrance = 0.25;
  double abl = 0.0;
  double gamut_p3 = 1.0;
  double emulate = 0.0;
  double sdr_white = 203.0;
  double em_base = 200.0;
  double em_pow = 6.0;
  double em_lo = 1.0;
  double em_hi = 1.0;
  double em_sat = 1.0;
  double black_lo = 0.0;
  double black_hi = 0.0;
  friend bool operator==(const GradeParams&, const GradeParams&) = default;
};

struct GradeDecision {
  GradeMode mode = GradeMode::Neutral;
  GradeParams params;  // significatifs hors Neutral
  // Où se place le blanc d'une source SDR dans une sortie PQ (hdr-reference-white de mpv) :
  // 203 nits avec l'émulation TrueHDR, le blanc papier sinon.
  double source_white_nits = grade_limits::kSdrWhiteNits;
};

// Le pic du grade : celui de l'écran s'il est connu (profil, système, --display), 1000 sinon.
// FieldOutOfRange hors de 100 à 10000 nits.
std::expected<double, Refusal> grade_peak(const core::DisplayInfo& display);

std::expected<GradeDecision, Refusal> decide_grade(const GradeSettings& settings,
                                                   const GradeContext& context);

}  // namespace ajp::render
