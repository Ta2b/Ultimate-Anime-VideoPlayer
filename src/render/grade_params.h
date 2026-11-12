// Réglages et paramètres du grade HDR, sans GPU.
//
// GradeSettings porte les réglages du grade (play.toml, section grade) ; decide_grade() en tire, pour une
// route, une source et un écran donnés, le mode du grade et les valeurs des //!PARAM des hooks
// (anime-hdr-params.hook). Le calcul part de M.plan de hdr-grade.lua
// (https://github.com/Ta2b/mpv-AnimeJaNai-HDR, commit 33ff79d) ; tests/unit/render/grade_params_test.cpp fixe
// ses sorties.
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string_view>

#include "core/profile.h"
#include "core/types.h"
#include "render/cube_lut.h"

namespace ajp::render {

// L'exposition par plan de la méthode anime : 0,25 diaphragme par diaphragme d'APL du plan rapportée à la
// médiane de la série, bornée à ±0,25 diaphragme (reprise de l'ancienne courbe par plan, étape 8.5).
inline constexpr double kAnimeExposureSlope = 0.25;
inline constexpr double kAnimeExposureMaxE = 0.25;

// Plages des réglages, incluses ; une valeur hors plage est refusée, jamais corrigée.
namespace grade_limits {
inline constexpr core::Range<double> kPaperWhite{80.0, 400.0};
inline constexpr core::Range<double> kShadows{0.0, 0.8};
inline constexpr core::Range<double> kHighlights{0.0, 1.0};
inline constexpr core::Range<double> kBlackNits{0.0, 50.0};
inline constexpr core::Range<double> kPeakNits{100.0, 10000.0};
// Vrai noir OLED : fin du pied, en pas de code de la source au-dessus du noir.
inline constexpr core::Range<double> kBlackSteps{0.5, 8.0};
inline constexpr core::Range<double> kAreaRtx{0.0, 1.0};
// Saturation, contraste et éclats de la méthode anime, en % : le menu, play.toml et le grade lisent la même
// plage.
inline constexpr core::Range<int> kAnimePercent{0, 200};
// Pic du grade quand l'écran ne le dit pas.
inline constexpr double kDefaultPeakNits = 1000.0;
// Le blanc d'une source SDR sur la route native : 203 nits (BT.2408).
inline constexpr double kSdrWhiteNits = 203.0;
}  // namespace grade_limits

// La lumière de la scène que le lecteur connaît (analyse à l'ingestion, pré-analyse des plans) : sa
// luminance moyenne au blanc SDR de 203 cd/m² et sa part de grands aplats clairs. Le grade en trois passes
// en tire son limiteur OLED et le haut de sa courbe (GradeParams::fs_apl, fs_bright).
inline constexpr double kAnimeLift = 1.15;  // GradeParams::an_lift, sans décision par plan

struct SceneLight {
  double apl_nits = 0.0;
  double lit = 0.0;
  friend bool operator==(const SceneLight&, const SceneLight&) = default;
};

// Les teintes de peau des visages du plan (media/faces.h, option face_skin) : teinte en degrés, saturation de
// 0 à 1, au plus trois. Aucune : la plage de peau large du grade (skin_like, 3-anime.hook).
struct SkinTones {
  std::array<double, 3> hue{};
  std::array<double, 3> sat{};
  int count = 0;
  friend bool operator==(const SkinTones&, const SkinTones&) = default;
};

struct GradeSettings {
  bool grade = true;              // grade activé
  bool grade_native_hdr = false;  // grade aussi une source déjà HDR
  // Route native (le défaut) : le blanc de la source entre dans le grade à 203 nits ; à l'écran, la pile de
  // LUT et la dernière passe le portent vers 70 % du pic, les éclats au-dessus (render/output_tune.h).
  // Faux : le blanc au blanc papier (paper_white). La clé de play.toml garde son nom d'avant (truehdr_look).
  bool truehdr_look = true;
  double paper_white = 203.0;  // nits, blanc papier
  double shadows = 0.15;
  // Éclats et grands aplats clairs : la montée au-dessus du blanc (spec_gain).
  double highlights_rtx = 0.25;
  double area_rtx = 0.0;
  // La lumière de la scène de l'image qui vient ; vide, inconnue. Jamais écrite dans play.toml.
  std::optional<SceneLight> scene_light;
  // La méthode « HDR anime » décidée par plan (méga plan A2, --anime-shots) : l'exposition des surfaces du
  // plan, en diaphragmes (|E| ≤ kAnimeExposureMaxE) ; 0 : le relèvement fixe. Jamais écrite dans play.toml.
  double anime_exposure = 0.0;
  // La médiane des luminances moyennes des plans connus (nits au blanc de 203 ; 0 : inconnue), pour les noirs
  // profonds (P3.52). Jamais écrite dans play.toml.
  double series_apl_nits = 0.0;
  // La peau des visages du plan (face_skin) ; jamais écrite dans play.toml.
  SkinTones skin_tones;
  // Le peps de la méthode « HDR anime » (--anime-contrast, --anime-saturation, --anime-highlights), en % ;
  // 100 : la méthode telle quelle. Écrit dans play.toml par le menu (settings_toml.cpp) ; une option donnée à
  // la main ne l'est pas (settings_to_save).
  int anime_contrast = 100;
  int anime_saturation = 100;
  int anime_highlights = 100;
  double chroma_base = 0.0;  // préréglage : saturation en plus (0 : aucune)
  double black_lo = 0.0;     // nits : vrai noir OLED sous ce seuil
  double black_hi = 0.0;     // nits : fin du pied ; <= black_lo, la passe est sautée
  bool true_black = false;   // vrai noir : passe black_lo / black_hi (OLED, ou forcé)
  friend bool operator==(const GradeSettings&, const GradeSettings&) = default;
};

// FieldOutOfRange en nommant le champ (« grade.paper_white »).
std::expected<void, Refusal> check_grade_settings(const GradeSettings& settings);

// Ce qui est gradé, et comment.
enum class GradeMode : std::uint8_t {
  Neutral,    // pas de grade : sortie non PQ, grade coupé, route Off ou source HDR non gradée
  Native,     // SDR converti par libplacebo en PQ (route native, ou mode SDR sur une sortie HDR10)
  NativeHdr,  // source déjà HDR, gradée parce que grade_native_hdr le demande
  Sdr,        // sortie SDR : couleurs, noirs, contraste et saturation, sans marge au-dessus du blanc
};

struct GradeContext {
  core::HdrMode route = core::HdrMode::Off;  // la route HDR jouée
  bool source_hdr = false;                   // source PQ ou HLG
  bool output_pq = false;                    // la cible est encodée en PQ ; sinon SDR BT.709
  bool enabled = true;                       // bascule A/B de l'utilisateur
  double peak_nits = grade_limits::kDefaultPeakNits;
};

// Les //!PARAM du grade (anime-hdr-params.hook).
struct GradeParams {
  double peak = 1000.0;
  double paper = 203.0;
  double shadows = 0.30;
  double shadow_knee = 50.0;
  double spec_gain = 4.3;
  // 1 : la source SDR est placée à sdr_white (route native, le défaut) ; 0 : son blanc au blanc papier.
  double adapt = 0.0;
  double sdr_white = 203.0;
  double black_lo = 0.0;
  double black_hi = 0.0;
  // Sortie SDR : 1, l'image de la cible est en BT.709 au gamma sdr_gamma, son blanc à `peak` nits.
  double sdr_out = 0.0;
  double sdr_gamma = 2.2;
  // Sortie SDR de la méthode « HDR anime » (sdr_tonemap) : 1, l'image SDR passe en PQ avant le grade, le
  // grade et la pile de LUT d'HDR travaillent comme sur un écran HDR de `peak` nits, puis une courbe par plan
  // ramène tm_white (les surfaces) au blanc SDR et tm_top (la marge des éclats du plan) juste sous lui.
  double sdr_tm = 0.0;
  double tm_white = 233.0;
  double tm_top = 1000.0;
  // Saturation en facteur (1 : identité) et pivot de la passe anime (nits).
  double chroma = 1.0;
  double pivot = 80.0;
  // Le look (LUT .cube, anime-hdr-look.hook) : intensité (0 : passe sautée), blanc de référence de la LUT en
  // nits, 1 si la LUT attend du PQ BT.2020 plutôt que du SDR BT.709.
  std::array<double, kLookSlots> look_mixes{};  // une intensité par LUT enchaînée, [0] la première
  double look_white = 203.0;
  double look_pq = 0.0;
  double look_clean = 0.0;  // 1 : la pile sur la couleur du voisinage, grain et paliers non étirés
  // Le blanc des surfaces en part de celui que la pile porte sous la courbe (1 : la pile seule ; 0,5 à 1),
  // réglé par la dernière passe (output_tune).
  double surface_white = 1.0;
  // Le pic d'avant le plafond des flashs dangereux (0 : celui de `peak`, rien n'est plafonné). Les
  // métadonnées HDR annoncées le gardent (B41-03) : elles ne changent pas à chaque entrée ou sortie de flash.
  double uncapped_peak = 0.0;
  // La luminance moyenne de la scène à l'écran (nits ; < 0 : inconnue), donnée par le lecteur.
  double fs_apl = -1.0;
  // La luminance moyenne de la série (la médiane des plans connus, nits ; < 0 : inconnue) : les noirs
  // profonds.
  double fs_series = -1.0;
  // Relèvement des surfaces (kAnimeLift, ajusté par plan : GradeSettings::anime_exposure).
  double an_lift = kAnimeLift;
  double an_punch = 0.0;   // --anime-contrast : contraste en plus (0 : aucun)
  double an_chroma = 1.0;  // --anime-saturation : facteur de saturation (1 : identité)
  double an_spec = 1.0;    // --anime-highlights : part de la montée des éclats (1 : celle du grade)
  // La peau des visages du plan (SkinTones) : skin_n teintes, 0 pour la plage large.
  double skin_n = 0.0;
  double skin_h0 = 0.0, skin_h1 = 0.0, skin_h2 = 0.0;
  double skin_s0 = 0.0, skin_s1 = 0.0, skin_s2 = 0.0;
  friend bool operator==(const GradeParams&, const GradeParams&) = default;
};

// Le grade de la chaîne pour chaque sortie ; l'écran prend celui de sa cible (PQ ou SDR).
struct GradeChoice {
  std::optional<GradeParams> pq;
  std::optional<GradeParams> sdr;
  // Sortie SDR de la méthode « HDR anime » : le grade PQ, ramené au SDR plan par plan (sdr_tonemap).
  std::optional<GradeParams> sdr_anime;
  friend bool operator==(const GradeChoice&, const GradeChoice&) = default;
};

struct GradeDecision {
  GradeMode mode = GradeMode::Neutral;
  GradeParams params;  // significatifs hors Neutral
  // Où se place le blanc d'une source SDR dans une sortie PQ (hdr-reference-white de mpv) :
  // 203 nits sur la route native (truehdr_look), le blanc papier sinon.
  double source_white_nits = grade_limits::kSdrWhiteNits;
};

// Le grade PQ d'une source SDR, tourné vers une sortie SDR (méthode « HDR anime ») : sdr_tm, les bornes de la
// courbe du plan (tm_white : les surfaces relevées ; tm_top : la marge des éclats que la méthode donne au
// plan, selon sa luminosité moyenne fs_apl), sans pile de LUT (étalonnée pour l'écran HDR). Le vrai noir
// reste : ses seuils sont mesurés sur la sortie jouée.
void sdr_tonemap(GradeParams& p);

// La sortie HDR d'une luminosité stable d'un plan à l'autre : le relèvement fixe des surfaces (sans
// exposition par plan) et la marge des éclats au pic, quelle que soit la clé de la scène (fs_apl inconnue) ;
// la dalle limite elle-même les scènes claires. La série (fs_series) reste : elle ne bouge pas aux coupes.
void steady_hdr(GradeParams& p);

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

}  // namespace ajp::render
