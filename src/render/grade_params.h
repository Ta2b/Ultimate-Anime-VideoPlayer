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
#include <optional>
#include <span>
#include <string_view>

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
inline constexpr double kSdkContrastMax = 200.0;
inline constexpr core::Range<double> kMiddleGray{10.0, 100.0};
inline constexpr core::Range<double> kSubtitleWhite{80.0, 1000.0};
inline constexpr core::Range<double> kBlackNits{0.0, 50.0};
inline constexpr core::Range<double> kPeakNits{100.0, 10000.0};
inline constexpr core::Range<double> kGuard{0.0, 1.0};
// Vrai noir OLED : fin du pied, en pas de code de la source au-dessus du noir.
inline constexpr core::Range<double> kBlackSteps{0.5, 8.0};
inline constexpr core::Range<double> kAreaRtx{0.0, 1.0};
inline constexpr core::Range<double> kExpand{0.0, 1.0};
// Seuils du grade adaptatif : luminance moyenne de l'image source (0 à 100 % du blanc), lissée sur une à deux
// secondes.
inline constexpr core::Range<double> kAdaptiveLuma{1.0, 99.0};
// Limiteur OLED : APL lissé (nits) où la modération des éclats commence, puis est complète.
inline constexpr core::Range<double> kAblNits{20.0, 2000.0};
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
  // Grade adaptatif : les curseurs (noirs, couleurs) et le gris moyen de RTX HDR sont ceux d'une scène sombre
  // ; ces valeurs, ceux d'une scène claire. On glisse de l'un à l'autre quand la luminance moyenne lissée de
  // l'image source passe de adaptive_lo à adaptive_hi (% du blanc, mesurée par le moteur TensorRT : sans lui,
  // aucun effet).
  bool adaptive = false;
  double adaptive_lo = 13.0;
  double adaptive_hi = 54.0;
  double adaptive_shadows = 0.80;
  double adaptive_vibrance = 0.0;
  double adaptive_gray = 30.0;
  double punch_base = 0.0;   // préréglage : contraste en plus, en sortie du grade (0 : aucun)
  double chroma_base = 0.0;  // préréglage : saturation en plus (0 : aucune)
  double middle_gray = 50.0;
  double subtitle_white = 200.0;  // nits, blanc des sous-titres en sortie HDR
  bool oled = false;              // dalle OLED : limiteur des scènes claires (abl)
  bool gamut_p3 = false;          // limite au gamut Display P3 de l'écran
  double black_lo = 0.0;          // nits : vrai noir OLED sous ce seuil
  double black_hi = 0.0;          // nits : fin du pied ; <= black_lo, la passe est sautée
  // Ajouts de ce dépôt (préréglage « Éclatant ») ; à 0, le grade est celui du lecteur actuel.
  double guard = 0.0;     // garde-fou des couleurs saturées près du pic
  double area_rtx = 0.0;  // TrueHDR et son émulation : gain des grands aplats clairs (0 : aucun)
  // Couleurs étendues : les couleurs BT.709 saturées vont vers les primaires P3 (peau et gris
  // épargnés), pour les dalles à large gamut comme le QD-OLED ; 0 : aucune.
  double expand = 0.0;
  bool true_black = false;  // vrai noir : passe black_lo / black_hi (OLED, ou forcé)
  // Limiteur OLED : APL de début et de fin de la modération ; 80 et 250 nits dans le lecteur actuel
  // (plein écran d'un QD-OLED en Peak 1000), tirés de la moyenne d'image de l'EDID sinon.
  double abl_lo = 80.0;
  double abl_hi = 250.0;
  friend bool operator==(const GradeSettings&, const GradeSettings&) = default;
};

// Préréglages du grade. Naturel : celui du lecteur actuel (GradeSettings{}). Éclatant : éclats,
// aplats lumineux, couleurs, couleurs étendues et noirs poussés, garde-fou des couleurs saturées.
enum class GradePreset : std::uint8_t { Eclatant, Naturel, Custom };
inline constexpr int kGradePresetCount = 3;
GradeSettings grade_preset(GradePreset preset, GradeSettings custom = {});
std::string_view preset_key(GradePreset preset);    // « eclatant », la clé de play.toml
std::string_view preset_label(GradePreset preset);  // « Vivid » : clé de traduction du menu (app/play/i18n.h)

// FieldOutOfRange en nommant le champ (« grade.paper_white ») ; abl_hi doit dépasser abl_lo.
std::expected<void, Refusal> check_grade_settings(const GradeSettings& settings);

// Ce qui est gradé, et comment.
enum class GradeMode : std::uint8_t {
  Neutral,    // pas de grade : sortie non PQ, grade coupé, route Off ou source HDR non gradée
  Native,     // SDR converti par libplacebo en PQ (route native), avec ou sans émulation TrueHDR
  TrueHdr,    // SDR passé par TrueHDR : l'image arrive déjà en PQ au pic de l'écran
  NativeHdr,  // source déjà HDR, gradée parce que grade_native_hdr le demande
  Sdr,        // sortie SDR : couleurs, noirs, contraste et saturation, sans marge au-dessus du blanc
};

struct GradeContext {
  core::HdrMode route = core::HdrMode::Off;  // celle du Plan
  bool source_hdr = false;                   // source PQ ou HLG
  bool output_pq = false;                    // la cible est encodée en PQ ; sinon SDR BT.709
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
  double guard = 0.0;
  double expand = 0.0;
  double abl_lo = 80.0;
  double abl_hi = 250.0;
  double gamut_p3 = 1.0;
  double emulate = 0.0;
  // HDR adaptatif (route native, tous GPU) : 1, l'aspect TrueHDR émulé suit la scène lissée.
  double adapt = 0.0;
  double sdr_white = 203.0;
  double em_base = 200.0;
  double em_pow = 6.0;
  double em_lo = 1.0;
  double em_hi = 1.0;
  double em_sat = 1.0;
  double black_lo = 0.0;
  double black_hi = 0.0;
  // Sortie SDR : 1, l'image de la cible est en BT.709 au gamma sdr_gamma, son blanc à `peak` nits.
  double sdr_out = 0.0;
  double sdr_gamma = 2.2;
  // Fin du grade (toutes routes) : contraste en courbe de puissance autour de `pivot` (nits), saturation en
  // facteur. 0 et 1 : identité exacte.
  double punch = 0.0;
  double chroma = 1.0;
  double pivot = 80.0;
  // Le look (LUT .cube, anime-hdr-look.hook) : intensité (0 : passe sautée), blanc de référence de la LUT en
  // nits, 1 si la LUT attend du PQ BT.2020 plutôt que du SDR BT.709.
  double look_mix = 0.0;
  double look_mix2 = 0.0;  // les LUT suivantes, enchaînées après la première (kLookSlots en tout)
  double look_mix3 = 0.0;
  double look_mix4 = 0.0;
  double look_mix5 = 0.0;
  double look_mix6 = 0.0;
  double look_mix7 = 0.0;
  double look_mix8 = 0.0;
  double look_mix9 = 0.0;
  double look_mix10 = 0.0;
  double look_mix11 = 0.0;
  double look_mix12 = 0.0;
  double look_mix13 = 0.0;
  double look_mix14 = 0.0;
  double look_mix15 = 0.0;
  double look_white = 203.0;
  double look_pq = 0.0;
  // Grade adaptatif des routes émulées (shader, tous GPU) : 1 pour l'activer, seuils en code PQ moyen de la
  // scène, valeurs d'une scène claire ; celles d'une scène sombre sont shadows, vibrance, em_base et em_pow.
  double ad_on = 0.0;
  double ad_lo = 0.2;
  double ad_hi = 0.45;
  double ad_shadows = 0.80;
  double ad_vibrance = 0.0;
  double ad_em_base = 120.0;
  double ad_em_pow = 6.0;
  friend bool operator==(const GradeParams&, const GradeParams&) = default;
};

// Le grade de la chaîne pour chaque sortie ; l'écran prend celui de sa cible (PQ ou SDR).
struct GradeChoice {
  std::optional<GradeParams> pq;
  std::optional<GradeParams> sdr;
  friend bool operator==(const GradeChoice&, const GradeChoice&) = default;
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

// Vrai noir OLED calé sur la rampe de 32 marches quasi noires de la source (8 bits limité : un pas
// vaut 1/219 de la plage), mesurée en nits à la sortie, grade compris, vrai noir coupé.
inline constexpr int kBlackRampSteps = 32;
struct BlackThresholds {
  double lo = 0.0;
  double hi = 0.0;
  friend bool operator==(const BlackThresholds&, const BlackThresholds&) = default;
};
// black_lo à mi-chemin du noir de la source (marche 0) et de la première marche, black_hi à la
// marche `steps` (interpolée) : le noir devient 0, la première marche reste visible, le pied
// rejoint l'identité à `steps` pas. Rien si la rampe ne monte pas (sortie plate, mesure ratée).
std::optional<BlackThresholds> black_thresholds(std::span<const double> step_nits, double steps);

// Plage du limiteur OLED tirée de la moyenne d'image maximale de l'écran (EDID, CTA-861) : la
// modération commence à 30 % de cette moyenne et devient complète à 94 % (80 et 250 nits pour
// 265 nits, les valeurs du lecteur actuel). Rien si la moyenne sort de la plage utile.
struct AblRange {
  double lo = 80.0;
  double hi = 250.0;
  friend bool operator==(const AblRange&, const AblRange&) = default;
};
std::optional<AblRange> abl_range(double frame_average_nits);

}  // namespace ajp::render
