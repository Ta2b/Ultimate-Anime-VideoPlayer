// Contrat du grade exact dans l'anneau (chantier 2c, docs/adr/0009-grade-dans-l-anneau.md) : les paramètres
// du grade fusionné (anime-hdr-fused/*.hook, anime-hdr-funcs.glsl), du look, du vrai noir et de KWin, la
// taille des cartes et les constantes du flou et du lissage de la zone.
//
// Les shaders de render/shaders restent la vérité de l'aspect : la référence kernels/ref/grade.cpp et les
// noyaux CUDA (gpu/cuda/backend/grade_kernels.cu) en sont des ports, et les formules d'un pixel sont écrites
// une seule fois, dans kernels/spec/grade_pixel.h, pour l'hôte comme pour le GPU.
//
// Cartes, comme libplacebo les pose sur OUTPUT (custom_mpv.c : taille arrondie au plus proche) :
// - locale (1-local.hook) : round(W / 8) × round(H / 8), trois composantes : la luminance en PQ, le « bleu »
//   max(0, B − max(R, V)) et la chroma max − min du SDR encodé, moyennes de 16 lectures bilinéaires du bloc ;
// - zone (2-area.hook) : round(Wl / 4) × round(Hl / 4) de la carte locale, moyenne 4 × 4 (4 lectures
//   bilinéaires), gaussienne séparable de rayon kBlurRadius (kBlurWeights, bords tenus), puis lissage
//   temporel : α = 1 − kSmoothBase^(kSmoothRate × Δt) quand la valeur gardée est à au plus kSmoothJump de la
//   mesure, sinon la mesure (une coupe repart de sa mesure) ; les kHistoryTexels premiers texels seulement ;
// - max local (2b-localmax.hook) : à la taille de la carte locale, le code de sortie gradé (look compris, ni
//   noir ni KWin) du pixel le plus clair (maxRGB du code PQ) du bloc 8 × 8 du texel, 0 sans vrai noir.
// Tests : tests/unit/kernels/grade_test.cpp, tests/unit/render/grade_ring_ref_test.cpp.
#pragma once

#include <array>
#include <cstddef>

namespace ajp::kernels {

namespace grade {

inline constexpr int kLocalDiv = 8;    // carte locale : 1/8 de l'image
inline constexpr int kAreaDiv = 4;     // zone : 1/4 de la carte locale (1/32 de l'image)
inline constexpr int kBlurRadius = 7;  // gaussienne σ = 3 texels de zone
inline constexpr std::array<float, 8> kBlurWeights{0.1346F, 0.1273F, 0.1078F, 0.0816F,
                                                   0.0553F, 0.0336F, 0.0182F, 0.0088F};
inline constexpr float kSmoothJump = 0.12F;  // écart en PQ au-delà duquel la zone repart de sa mesure
inline constexpr float kSmoothBase = 0.7F;   // α = 1 − 0,7^(48 Δt) : ~0,1 s quelle que soit la cadence
inline constexpr float kSmoothRate = 48.0F;
inline constexpr int kHistoryTexels = 32768;    // l'historique du shader (AJG_AREA_STATE)
inline constexpr int kSkinStep = 2;             // masque peau : moyenne 3 × 3 de pas 2 px
inline constexpr int kInputTablePoints = 4096;  // entrée SDR → linéaire, par canal

// Taille d'une carte comme libplacebo la pose : roundf(size / div), au moins 1.
constexpr int map_size(int size, int div) {
  const int out = (2 * size + div) / (2 * div);  // arrondi au plus proche, les moitiés vers le haut
  return out < 1 ? 1 : out;
}

}  // namespace grade

// Les //!PARAM lus par le grade fusionné, le look (pile composée en un seul emplacement), le vrai noir et
// KWin, sous leurs noms ; défauts de anime-hdr-params.hook et anime-hdr-look.hook.
struct GradeUniforms {
  float peak = 1000.0F;
  float paper = 203.0F;
  float in_max = 203.0F;
  float shadows = 0.30F;
  float shadow_knee = 50.0F;
  float area_gain = 1.4F;
  float spec_gain = 4.3F;
  float vibrance = 0.25F;
  float abl = 0.0F;
  float abl_lo = 80.0F;
  float abl_hi = 250.0F;
  float expand = 0.0F;
  float gamut_p3 = 1.0F;
  float guard = 0.0F;
  float emulate = 0.0F;
  float sdr_white = 203.0F;
  float em_base = 200.0F;
  float em_pow = 6.0F;
  float em_lo = 1.0F;
  float em_hi = 1.0F;
  float em_sat = 1.0F;
  float sdr_out = 0.0F;
  float sdr_gamma = 2.2F;
  float adapt = 0.0F;
  float punch = 0.0F;
  float chroma = 1.0F;
  float pivot = 80.0F;
  float ad_on = 0.0F;
  float ad_lo = 0.2F;
  float ad_shadows = 0.80F;
  float ad_vibrance = 0.0F;
  float ad_em_base = 120.0F;
  float ad_em_pow = 6.0F;
  float ad_sky_vibrance = 0.0F;
  float gl_force = -1.0F;
  float gl_shadows = 0.20F;
  float gl_vibrance = 0.05F;
  float gl_em_base = 160.0F;
  float gl_em_pow = 6.0F;
  float sh_on = 0.0F;
  float sh_t = 0.0F;
  float sh_glow = 0.0F;
  float sh_sky = 0.0F;
  float sh_series = 0.0F;
  float fs_apl = -1.0F;
  float fs_bright = 0.0F;
  float black_lo = 0.0F;
  float black_hi = 0.0F;
  float kwin_on = 0.0F;
  float kwin_ref = 200.0F;
  float kwin_peak = 1000.0F;
  float look_mix = 0.0F;  // la pile composée : 1 quand une LUT au moins est active, 0 sinon
  float look_white = 203.0F;
  float look_pq = 0.0F;
  float look_gamma = 2.4F;
  // Faux : le masque peau est coupé (sans --skin-mask, skin_area rend 0).
  float skin_on = 1.0F;

  friend bool operator==(const GradeUniforms&, const GradeUniforms&) = default;
};

// Les champs de GradeUniforms sous le nom de leur //!PARAM (skin_on n'en est pas un).
using GradeUniformMember =
    float GradeUniforms::*;  // l'alias évite un faux « unnecessary parentheses » (nvcc, hôte)
struct GradeUniformField {
  const char* name;
  GradeUniformMember field;
};
inline constexpr std::array<GradeUniformField, 55> kGradeUniformFields{{
    {"peak", &GradeUniforms::peak},
    {"paper", &GradeUniforms::paper},
    {"in_max", &GradeUniforms::in_max},
    {"shadows", &GradeUniforms::shadows},
    {"shadow_knee", &GradeUniforms::shadow_knee},
    {"area_gain", &GradeUniforms::area_gain},
    {"spec_gain", &GradeUniforms::spec_gain},
    {"vibrance", &GradeUniforms::vibrance},
    {"abl", &GradeUniforms::abl},
    {"abl_lo", &GradeUniforms::abl_lo},
    {"abl_hi", &GradeUniforms::abl_hi},
    {"expand", &GradeUniforms::expand},
    {"gamut_p3", &GradeUniforms::gamut_p3},
    {"guard", &GradeUniforms::guard},
    {"emulate", &GradeUniforms::emulate},
    {"sdr_white", &GradeUniforms::sdr_white},
    {"em_base", &GradeUniforms::em_base},
    {"em_pow", &GradeUniforms::em_pow},
    {"em_lo", &GradeUniforms::em_lo},
    {"em_hi", &GradeUniforms::em_hi},
    {"em_sat", &GradeUniforms::em_sat},
    {"sdr_out", &GradeUniforms::sdr_out},
    {"sdr_gamma", &GradeUniforms::sdr_gamma},
    {"adapt", &GradeUniforms::adapt},
    {"punch", &GradeUniforms::punch},
    {"chroma", &GradeUniforms::chroma},
    {"pivot", &GradeUniforms::pivot},
    {"ad_on", &GradeUniforms::ad_on},
    {"ad_lo", &GradeUniforms::ad_lo},
    {"ad_shadows", &GradeUniforms::ad_shadows},
    {"ad_vibrance", &GradeUniforms::ad_vibrance},
    {"ad_em_base", &GradeUniforms::ad_em_base},
    {"ad_em_pow", &GradeUniforms::ad_em_pow},
    {"ad_sky_vibrance", &GradeUniforms::ad_sky_vibrance},
    {"gl_force", &GradeUniforms::gl_force},
    {"gl_shadows", &GradeUniforms::gl_shadows},
    {"gl_vibrance", &GradeUniforms::gl_vibrance},
    {"gl_em_base", &GradeUniforms::gl_em_base},
    {"gl_em_pow", &GradeUniforms::gl_em_pow},
    {"sh_on", &GradeUniforms::sh_on},
    {"sh_t", &GradeUniforms::sh_t},
    {"sh_glow", &GradeUniforms::sh_glow},
    {"sh_sky", &GradeUniforms::sh_sky},
    {"sh_series", &GradeUniforms::sh_series},
    {"fs_apl", &GradeUniforms::fs_apl},
    {"fs_bright", &GradeUniforms::fs_bright},
    {"black_lo", &GradeUniforms::black_lo},
    {"black_hi", &GradeUniforms::black_hi},
    {"kwin_on", &GradeUniforms::kwin_on},
    {"kwin_ref", &GradeUniforms::kwin_ref},
    {"kwin_peak", &GradeUniforms::kwin_peak},
    {"look_mix", &GradeUniforms::look_mix},
    {"look_white", &GradeUniforms::look_white},
    {"look_pq", &GradeUniforms::look_pq},
    {"look_gamma", &GradeUniforms::look_gamma},
}};

}  // namespace ajp::kernels
