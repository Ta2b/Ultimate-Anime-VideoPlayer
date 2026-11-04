// Les formules du grade exact, écrites une fois pour l'hôte (kernels/ref/grade.cpp) et le GPU
// (gpu/cuda/backend/grade_kernels.cu) : un port ligne à ligne de anime-hdr-funcs.glsl, 1-local.hook,
// 2-area.hook, 2b-localmax.hook, anime-hdr-look.hook (ajl_look) et anime-hdr-black.hook (ajb_black,
// ajk_kwin). Mêmes opérations dans le même ordre que le GLSL ; mix(a, b, t) = a (1 − t) + b t, smoothstep et
// step comme en GLSL, les matrices recopiées colonne par colonne comme dans les shaders. Contrat :
// kernels/spec/grade.h.
//
// Les lectures d'image passent par des fonctions données par l'appelant (Fetch : code PQ d'un pixel aux
// indices bornés ; Look : lecture trilinéaire de la pile de LUT), pour que le CUDA y mette ses textures.
#pragma once

#include <cmath>

#include "kernels/spec/grade.h"

#if defined(__CUDACC__)
#define AJP_GRADE_FN __host__ __device__ inline
#else
#define AJP_GRADE_FN inline
#endif

namespace ajp::kernels::grade {

struct V3 {
  float r;
  float g;
  float b;
};

AJP_GRADE_FN V3 v3(float s) {
  return {s, s, s};
}
AJP_GRADE_FN V3 operator+(V3 a, V3 b) {
  return {a.r + b.r, a.g + b.g, a.b + b.b};
}
AJP_GRADE_FN V3 operator-(V3 a, V3 b) {
  return {a.r - b.r, a.g - b.g, a.b - b.b};
}
AJP_GRADE_FN V3 operator*(V3 a, V3 b) {
  return {a.r * b.r, a.g * b.g, a.b * b.b};
}
AJP_GRADE_FN V3 operator*(V3 a, float s) {
  return {a.r * s, a.g * s, a.b * s};
}
AJP_GRADE_FN V3 operator*(float s, V3 a) {
  return {s * a.r, s * a.g, s * a.b};
}
AJP_GRADE_FN V3 operator/(V3 a, float s) {
  return {a.r / s, a.g / s, a.b / s};
}
AJP_GRADE_FN V3 operator/(V3 a, V3 b) {
  return {a.r / b.r, a.g / b.g, a.b / b.b};
}

AJP_GRADE_FN float gmin(float a, float b) {
  return b < a ? b : a;
}
AJP_GRADE_FN float gmax(float a, float b) {
  return a < b ? b : a;
}
AJP_GRADE_FN float gclamp(float x, float lo, float hi) {
  return gmin(gmax(x, lo), hi);
}
AJP_GRADE_FN V3 vmax(V3 a, float s) {
  return {gmax(a.r, s), gmax(a.g, s), gmax(a.b, s)};
}
AJP_GRADE_FN V3 vclamp(V3 a, float lo, float hi) {
  return {gclamp(a.r, lo, hi), gclamp(a.g, lo, hi), gclamp(a.b, lo, hi)};
}
AJP_GRADE_FN V3 vpow(V3 a, float e) {
  return {powf(a.r, e), powf(a.g, e), powf(a.b, e)};
}
AJP_GRADE_FN float dot3(V3 a, V3 b) {
  return a.r * b.r + a.g * b.g + a.b * b.b;
}
AJP_GRADE_FN float max3(V3 v) {
  return gmax(v.r, gmax(v.g, v.b));
}
AJP_GRADE_FN float min3(V3 v) {
  return gmin(v.r, gmin(v.g, v.b));
}
AJP_GRADE_FN float mixf(float a, float b, float t) {
  return a * (1.0F - t) + b * t;
}
AJP_GRADE_FN V3 vmix(V3 a, V3 b, float t) {
  return a * (1.0F - t) + b * t;
}
AJP_GRADE_FN float smooth(float e0, float e1, float x) {
  const float t = gclamp((x - e0) / (e1 - e0), 0.0F, 1.0F);
  return t * t * (3.0F - 2.0F * t);
}
AJP_GRADE_FN float stepf(float edge, float x) {
  return x < edge ? 0.0F : 1.0F;
}
AJP_GRADE_FN float channel(V3 v, int i) {
  return i == 0 ? v.r : (i == 1 ? v.g : v.b);
}

// Matrice GLSL mat3(c0.x, c0.y, c0.z, c1.x, ...) : colonne par colonne.
struct M3 {
  float c[9];
};
AJP_GRADE_FN V3 mul(const M3& m, V3 v) {
  return {m.c[0] * v.r + m.c[3] * v.g + m.c[6] * v.b, m.c[1] * v.r + m.c[4] * v.g + m.c[7] * v.b,
          m.c[2] * v.r + m.c[5] * v.g + m.c[8] * v.b};
}
AJP_GRADE_FN M3 bt2020_to_p3() {
  return {{1.3435783F, -0.0652975F, 0.0028218F, -0.2821797F, 1.0757879F, -0.0195985F, -0.0613986F,
           -0.0104905F, 1.0167767F}};
}
AJP_GRADE_FN M3 p3_to_bt2020() {
  return {{0.7538330F, 0.0457438F, -0.0012103F, 0.1985974F, 0.9417772F, 0.0176017F, 0.0475696F, 0.0124789F,
           0.9836086F}};
}
AJP_GRADE_FN M3 bt2020_to_709() {
  return {{1.6604910F, -0.1245505F, -0.0181508F, -0.5876411F, 1.1328999F, -0.1005789F, -0.0728499F,
           -0.0083494F, 1.1187297F}};
}
AJP_GRADE_FN M3 bt709_to_2020() {
  return {{0.6274040F, 0.0690970F, 0.0163916F, 0.3292820F, 0.9195400F, 0.0880132F, 0.0433136F, 0.0113612F,
           0.8955950F}};
}
AJP_GRADE_FN M3 rgb_to_lms() {
  return {{1688.0F / 4096.0F, 683.0F / 4096.0F, 99.0F / 4096.0F, 2146.0F / 4096.0F, 2951.0F / 4096.0F,
           309.0F / 4096.0F, 262.0F / 4096.0F, 462.0F / 4096.0F, 3688.0F / 4096.0F}};
}
AJP_GRADE_FN M3 lms_to_rgb() {
  return {{3.4366067F, -0.7913296F, -0.0259499F, -2.5064521F, 1.9836005F, -0.0989137F, 0.0698454F,
           -0.1922709F, 1.1248636F}};
}
AJP_GRADE_FN M3 lms_to_ictcp() {
  return {{2048.0F / 4096.0F, 6610.0F / 4096.0F, 17933.0F / 4096.0F, 2048.0F / 4096.0F, -13613.0F / 4096.0F,
           -17390.0F / 4096.0F, 0.0F, 7003.0F / 4096.0F, -543.0F / 4096.0F}};
}
AJP_GRADE_FN M3 ictcp_to_lms() {
  return {{1.0F, 1.0F, 1.0F, 0.0086090F, -0.0086090F, 0.5600313F, 0.1110296F, -0.1110296F, -0.3206272F}};
}
// inverse() de GLSL, par la comatrice, en float.
AJP_GRADE_FN M3 inverse(const M3& m) {
  const float a00 = m.c[0], a10 = m.c[1], a20 = m.c[2];
  const float a01 = m.c[3], a11 = m.c[4], a21 = m.c[5];
  const float a02 = m.c[6], a12 = m.c[7], a22 = m.c[8];
  const float b00 = a11 * a22 - a12 * a21;
  const float b01 = a02 * a21 - a01 * a22;
  const float b02 = a01 * a12 - a02 * a11;
  const float b10 = a12 * a20 - a10 * a22;
  const float b11 = a00 * a22 - a02 * a20;
  const float b12 = a02 * a10 - a00 * a12;
  const float b20 = a10 * a21 - a11 * a20;
  const float b21 = a01 * a20 - a00 * a21;
  const float b22 = a00 * a11 - a01 * a10;
  const float det = a00 * b00 + a01 * b10 + a02 * b20;
  return {
      {b00 / det, b10 / det, b20 / det, b01 / det, b11 / det, b21 / det, b02 / det, b12 / det, b22 / det}};
}

AJP_GRADE_FN V3 pq_to_nits(V3 e) {
  const V3 p = vpow(vclamp(e, 0.0F, 1.0F), 1.0F / 78.84375F);
  const V3 num = vmax(p - v3(0.8359375F), 0.0F);
  const V3 den = vmax(v3(18.8515625F) - 18.6875F * p, 1e-6F);
  return 10000.0F * vpow(num / den, 6.27739463F);
}
AJP_GRADE_FN V3 nits_to_pq(V3 n) {
  const V3 q = vpow(vclamp(n / 10000.0F, 0.0F, 1.0F), 0.1593017578125F);
  return vpow((v3(0.8359375F) + 18.8515625F * q) / (v3(1.0F) + 18.6875F * q), 78.84375F);
}

// L'entrée du ring (GradeInput, kernels/ref/grade.h) : `nits` les trois tables de kInputTablePoints
// luminances mises bout à bout (R, G, B), lues par interpolation linéaire, puis la matrice `m` (ligne par
// ligne) et le PQ.
AJP_GRADE_FN float input_lookup(const float* table, float code) {
  const float x = gclamp(code, 0.0F, 1.0F) * static_cast<float>(kInputTablePoints - 1);
  int i = static_cast<int>(x);
  i = i > kInputTablePoints - 2 ? kInputTablePoints - 2 : i;
  const float t = x - static_cast<float>(i);
  return table[i] * (1.0F - t) + table[i + 1] * t;
}
AJP_GRADE_FN V3 input_code(V3 code, const float* nits, const float* m) {
  const float r = input_lookup(nits, code.r);
  const float g = input_lookup(nits + kInputTablePoints, code.g);
  const float b = input_lookup(nits + 2 * kInputTablePoints, code.b);
  const V3 lin{m[0] * r + m[1] * g + m[2] * b, m[3] * r + m[4] * g + m[5] * b,
               m[6] * r + m[7] * g + m[8] * b};
  return nits_to_pq(vmax(lin, 0.0F));
}

AJP_GRADE_FN float scene_apl(const GradeUniforms& u) {
  return u.fs_apl >= 0.0F ? u.fs_apl : 0.15F * u.sdr_white;
}

// L'image de la cible en lumière linéaire BT.2020 (nits), et retour : PQ, ou SDR BT.709.
AJP_GRADE_FN V3 decode(V3 e, const GradeUniforms& u) {
  if (u.sdr_out > 0.5F) return mul(bt709_to_2020(), vpow(vclamp(e, 0.0F, 1.0F), u.sdr_gamma) * u.peak);
  return pq_to_nits(e);
}
AJP_GRADE_FN V3 encode(V3 n, const GradeUniforms& u) {
  if (u.sdr_out > 0.5F) return vpow(vclamp(mul(bt2020_to_709(), n) / u.peak, 0.0F, 1.0F), 1.0F / u.sdr_gamma);
  return nits_to_pq(n);
}
AJP_GRADE_FN V3 to_target(V3 bt2020, const GradeUniforms& u) {
  if (u.sdr_out > 0.5F) return mul(bt2020_to_709(), bt2020);
  return u.gamut_p3 > 0.5F ? mul(bt2020_to_p3(), bt2020) : bt2020;
}
AJP_GRADE_FN V3 from_target(V3 c, const GradeUniforms& u) {
  if (u.sdr_out > 0.5F) return mul(bt709_to_2020(), c);
  return u.gamut_p3 > 0.5F ? mul(p3_to_bt2020(), c) : c;
}
AJP_GRADE_FN V3 target_luma(const GradeUniforms& u) {
  if (u.sdr_out > 0.5F) return {0.2126F, 0.7152F, 0.0722F};
  return u.gamut_p3 > 0.5F ? V3{0.2289746F, 0.6917385F, 0.0792869F} : V3{0.2627F, 0.6780F, 0.0593F};
}

AJP_GRADE_FN float shadow_ratio(float m, float depth, const GradeUniforms& u) {
  if (m >= u.shadow_knee) return 1.0F;
  const float t = 1.0F - m / u.shadow_knee;
  return 1.0F - depth * t * t;
}

AJP_GRADE_FN V3 to_ictcp(V3 bt2020_nits) {
  return mul(lms_to_ictcp(), nits_to_pq(mul(rgb_to_lms(), vmax(bt2020_nits, 0.0F))));
}

AJP_GRADE_FN float skin_mask(V3 ictcp) {
  const float chroma = sqrtf(ictcp.g * ictcp.g + ictcp.b * ictcp.b);
  const float hue = atan2f(ictcp.b, ictcp.g) * (180.0F / 3.14159265358979F);
  return smooth(88.0F, 102.0F, hue) * (1.0F - smooth(140.0F, 154.0F, hue)) * smooth(0.008F, 0.018F, chroma) *
         (1.0F - smooth(0.095F, 0.13F, chroma));
}

// Le masque peau d'un pixel : moyenne 3 × 3, pas de kSkinStep px, des masques des codes voisins.
// fetch(x, y) rend le code du pixel aux indices bornés à l'image.
template <class Fetch>
AJP_GRADE_FN float skin_area(const Fetch& fetch, int x, int y, const GradeUniforms& u) {
  if (u.skin_on < 0.5F) return 0.0F;
  float acc = 0.0F;
  for (int j = -1; j <= 1; j++)
    for (int i = -1; i <= 1; i++)
      acc += skin_mask(to_ictcp(decode(fetch(x + i * kSkinStep, y + j * kSkinStep), u)));
  return acc / 9.0F;
}

AJP_GRADE_FN float skin_ratio(float y, float skin, const GradeUniforms& u) {
  if (u.sdr_out > 0.5F || skin <= 0.0F) return 1.0F;
  const float k = gclamp(0.15F * u.peak, 100.0F, 200.0F);
  if (y <= k) return 1.0F;
  return mixf(1.0F, sqrtf(k / y), skin);
}

AJP_GRADE_FN float highlight_ratio(float y, float isolation, float skin, float abl_scale,
                                   const GradeUniforms& u) {
  float gain = mixf(mixf(u.area_gain, u.spec_gain, isolation), u.area_gain, skin);
  gain = 1.0F + (gain - 1.0F) * abl_scale;
  float out_y = y * (1.0F + smooth(0.5F * u.paper, u.paper, y) * (gain - 1.0F));
  const float top = u.in_max * (1.0F + smooth(0.5F * u.paper, u.paper, u.in_max) * (gain - 1.0F));
  const float knee = 0.65F * u.peak;
  if (top > u.peak && out_y > knee) {
    const float z = (out_y - knee) / (u.peak - knee);
    const float zm = (top - knee) / (u.peak - knee);
    out_y = knee + (u.peak - knee) * z * (1.0F + z / (zm * zm)) / (1.0F + z);
  }
  return out_y / y;
}

AJP_GRADE_FN V3 truehdr_look(V3 nits, float em_base, float em_pow, const GradeUniforms& u) {
  const V3 luma{0.2627F, 0.6780F, 0.0593F};
  float top = u.peak;
  float lo = u.em_lo;
  if (u.adapt > 0.5F) {
    const float key = gclamp(scene_apl(u) / u.sdr_white, 0.0F, 1.0F);
    const float bright = gclamp(u.fs_bright, 0.0F, 1.0F);
    const float g =
        mixf(1.0F, 0.55F, smooth(0.30F, 0.70F, key)) * mixf(1.0F, 0.75F, smooth(0.05F, 0.30F, bright));
    top = gmax(u.peak * g, gmin(u.peak, 1.5F * em_base));
    lo = mixf(1.0F + 0.5F * (u.em_lo - 1.0F), u.em_lo, smooth(0.04F, 0.15F, key));
  }
  const V3 x = vmax(nits, 0.0F) / u.sdr_white;
  const float Y = dot3(x, luma);
  const float idx = gclamp(Y + 0.15F * (gmax(x.r, gmax(x.g, x.b)) - Y), 0.0F, 1.0F);
  if (idx < 1e-6F) return v3(0.0F);
  const float s2 = powf(idx, 2.0F / 2.4F);
  V3 c = x * (em_base * s2 / idx * expf(logf(gmax(top / em_base, 1.0F)) * powf(s2, 0.5F * em_pow)));
  const float Yn = gmax(dot3(c, luma), 1e-6F);
  const float P = 0.2575F * em_base;
  float Y2;
  if (Yn < P) {
    Y2 = P * powf(Yn / P, lo);
  } else {
    const float span = logf(top / P);
    const float t = gmin(logf(Yn / P) / span, 1.0F);
    Y2 = P * expf(t * (u.em_hi - (u.em_hi - 1.0F) * t) * span);
  }
  c = c * (Y2 / Yn);
  return vmax(v3(Y2) + u.em_sat * (c - v3(Y2)), 0.0F);
}

AJP_GRADE_FN float gamut_room(V3 a, V3 b, const GradeUniforms& u) {
  const V3 d = b - a;
  float t = 1.0F;
  for (int i = 0; i < 3; i++) {
    const float di = channel(d, i);
    const float ai = channel(a, i);
    if (di < -1e-6F)
      t = gmin(t, gmax(ai, 0.0F) / -di);
    else if (di > 1e-6F)
      t = gmin(t, gmax(u.peak - ai, 0.0F) / di);
  }
  return t;
}

// ajg_grade : `lin` en nits BT.2020 linéaires, `skin_sm` la peau du voisinage, `local_raw` la carte locale
// (composante x) et `area_pq` la zone à sa place. Rend la couleur gradée, en nits BT.2020 linéaires.
AJP_GRADE_FN V3 grade_linear(V3 lin, float skin_sm, float local_raw, float area_pq, const GradeUniforms& u) {
  float t_scene = 0.0F;
  float glow = 0.0F;
  float sky = 0.0F;
  float series_bright = 0.0F;
  if (u.ad_on > 0.5F && u.gl_force >= 0.0F) {
    t_scene = u.ad_lo < 0.5F ? 1.0F : 0.0F;
    glow = gclamp(u.gl_force, 0.0F, 1.0F);
  } else if (u.ad_on > 0.5F && u.sh_on > 0.5F) {
    t_scene = u.sh_t;
    glow = u.sh_glow;
    sky = u.sh_sky;
    series_bright = u.sh_series;
  }
  const float sh_now = mixf(mixf(u.shadows, u.ad_shadows, t_scene), u.gl_shadows, glow);
  const float vb_now =
      mixf(mixf(mixf(u.vibrance, u.ad_vibrance, t_scene), u.gl_vibrance, glow), u.ad_sky_vibrance, sky);
  const float punch_now = u.punch + 0.12F * series_bright;
  const float chroma_now = u.chroma + 0.06F * series_bright;
  const float em_base_now = mixf(mixf(u.em_base, u.ad_em_base, t_scene), u.gl_em_base, glow);
  const float em_pow_now = mixf(mixf(u.em_pow, u.ad_em_pow, t_scene), u.gl_em_pow, glow);
  const float y_src = dot3(lin, V3{0.2627F, 0.6780F, 0.0593F});
  if (u.emulate > 0.5F) {
    const float skin_src = (u.ad_on > 0.5F) ? skin_sm : 0.0F;
    V3 looked = truehdr_look(lin, em_base_now, em_pow_now, u);
    if (skin_src > 0.0F) {
      const float kSkinAnchor = 0.6F;
      const V3 anchored =
          truehdr_look(lin, 0.5F * (u.em_base + u.ad_em_base), 0.5F * (u.em_pow + u.ad_em_pow), u);
      looked = vmix(looked, anchored, kSkinAnchor * skin_src);
    }
    lin = looked;
  }
  const bool p3 = u.gamut_p3 > 0.5F;
  V3 c = to_target(lin, u);
  if (u.emulate > 0.5F) c = vmax(c, 0.0F);

  float m = max3(c);
  float y = dot3(c, target_luma(u));
  const float y_pq = nits_to_pq(v3(gmax(y_src, 0.0F))).r;
  const float local_pq = gmax(local_raw, y_pq - 0.08F);
  float isolation = smooth(0.05F, 0.25F, local_pq - area_pq);
  const float abl_scale = u.fs_apl >= 0.0F ? 1.0F - u.abl * smooth(u.abl_lo, u.abl_hi, u.fs_apl) : 1.0F;
  V3 ictcp_in = to_ictcp(lin);
  const float skin = skin_sm;
  float tame = 0.0F;
  if (m > 1e-4F && u.sdr_out < 0.5F) {
    const float sat_px = 1.0F - gclamp(min3(c) / m, 0.0F, 1.0F);
    tame = smooth(0.45F, 0.85F, sat_px) * smooth(0.12F * u.peak, 0.30F * u.peak, m) *
           smooth(0.46F, 0.55F, area_pq) * (1.0F - skin);
    isolation *= 1.0F - tame;
  }
  if (u.expand > 0.0F) {
    const V3 c709 = mul(bt2020_to_709(), lin);
    const float hi = max3(c709);
    if (hi > 1e-4F) {
      const float sat = 1.0F - gclamp(min3(c709) / hi, 0.0F, 1.0F);
      const float w = u.expand * smooth(0.08F, 0.45F, sat) * (1.0F - skin) * (1.0F - tame) *
                      smooth(0.12F, 0.30F, ictcp_in.r);
      if (w > 0.0F) {
        const V3 wide = mul(p3_to_bt2020(), vmax(c709, 0.0F));
        const V3 base = p3 ? mul(bt2020_to_p3(), lin) : lin;
        const V3 target = p3 ? mul(bt2020_to_p3(), wide) : wide;
        c = base + (target - base) * (w * gamut_room(base, target, u));
        if (u.emulate > 0.5F) c = vmax(c, 0.0F);
        lin = p3 ? mul(p3_to_bt2020(), c) : c;
        ictcp_in = to_ictcp(lin);
        m = max3(c);
        y = dot3(c, p3 ? V3{0.2289746F, 0.6917385F, 0.0792869F} : V3{0.2627F, 0.6780F, 0.0593F});
      }
    }
  }
  float r = 1.0F;
  if (m > 1e-4F && y > 1e-4F) {
    r = shadow_ratio(m, sh_now, u);
    r *= highlight_ratio(y * r, isolation, skin, abl_scale, u);
    r *= skin_ratio(y * r, skin, u);
    if (u.guard > 0.0F || tame > 0.0F) {
      const float sat = 1.0F - min3(c) / m;
      const float w = gmax(u.guard * smooth(0.25F, 0.75F, sat), tame);
      const float knee = mixf(0.7F, 0.5F, tame) * u.peak;
      const float mo = m * r;
      if (w > 0.0F && mo > knee) {
        const float z = (mo - knee) / (u.peak - knee);
        r *= mixf(1.0F, (knee + (u.peak - knee) * z / (1.0F + z)) / mo, w);
      }
    }
    r = gmin(r, u.peak / m);
    c = c * r;
  }

  if (vb_now > 0.0F) {
    V3 ictcp = fabsf(r - 1.0F) < 1e-5F ? ictcp_in : to_ictcp(from_target(c, u));
    const float chroma = sqrtf(ictcp.g * ictcp.g + ictcp.b * ictcp.b);
    const float weight = smooth(0.0F, 0.06F, chroma) * (1.0F - smooth(0.10F, 0.40F, chroma)) *
                         smooth(0.12F, 0.30F, ictcp.r) * (1.0F - skin) * (1.0F - tame);
    const float warm = smooth(0.0F, 0.01F, ictcp.b - 0.5F * gmax(ictcp.g, 0.0F));
    const float vivid_cool = gmax(smooth(0.04F, 0.10F, chroma), smooth(0.46F, 0.54F, ictcp.r));
    const float vb_px = mixf(vb_now, gmin(vb_now, 0.10F), gmax(warm, vivid_cool));
    const float boost = 1.0F + vb_px * weight;
    ictcp.g *= boost;
    ictcp.b *= boost;
    const V3 vivid = to_target(mul(lms_to_rgb(), pq_to_nits(mul(ictcp_to_lms(), ictcp))), u);
    const float room = gamut_room(c, vivid, u);
    c = vmix(c, vivid, room / powf(1.0F + powf(room, 4.0F), 0.25F));
  }

  if (punch_now != 0.0F || chroma_now != 1.0F) {
    float yt = dot3(c, target_luma(u));
    if (yt > 1e-4F) {
      if (punch_now != 0.0F) {
        const float x = yt / u.pivot;
        const float expo = 1.0F + punch_now * (x >= 1.0F ? 0.55F : 0.40F);
        float shaped = u.pivot * powf(x, expo);
        const float knee = 0.75F * u.peak;
        if (shaped > knee) {
          const float z = (shaped - knee) / (u.peak - knee);
          shaped = knee + (u.peak - knee) * z / (1.0F + z);
        }
        const float ratio = shaped / yt;
        c = c * ratio;
        yt *= ratio;
      }
      if (chroma_now != 1.0F) {
        const float k = 1.0F + (chroma_now - 1.0F) * (1.0F - 0.6F * smooth(0.35F * u.peak, u.peak, yt)) *
                                   (1.0F - tame * stepf(1.0F, chroma_now));
        const V3 grey = v3(yt);
        const V3 target = grey + (c - grey) * k;
        c = grey + (target - grey) * gamut_room(grey, target, u);
      }
      c = vclamp(c, 0.0F, u.peak);
    }
  }
  return from_target(c, u);
}

// ajg_encode(ajg_grade(ajg_decode(e), ...)) : le code de sortie gradé, avant le look.
AJP_GRADE_FN V3 grade_code(V3 e, float skin_sm, float local_raw, float area_pq, const GradeUniforms& u) {
  return encode(grade_linear(decode(e, u), skin_sm, local_raw, area_pq, u), u);
}

// La pile de LUT d'un emplacement : lut(pos) rend la lecture trilinéaire en pos ∈ [0, 1]³ (centres des texels
// aux bords, comme le hook : pos = (c (n − 1) + 0,5) / n sur une texture de n points).
template <class Lut>
AJP_GRADE_FN V3 look_chain(V3 c, const Lut& lut, const GradeUniforms& u) {
  if (u.look_mix > 0.0F) c = vmix(c, vclamp(lut(vclamp(c, 0.0F, 1.0F)), 0.0F, 1.0F), u.look_mix);
  return c;
}

AJP_GRADE_FN V3 look_decode(V3 e, const GradeUniforms& u) {
  if (u.sdr_out > 0.5F) return vpow(vclamp(e, 0.0F, 1.0F), u.sdr_gamma) * u.peak;
  return pq_to_nits(e);
}
AJP_GRADE_FN V3 look_encode(V3 n, const GradeUniforms& u) {
  if (u.sdr_out > 0.5F) return vpow(vclamp(n / u.peak, 0.0F, 1.0F), 1.0F / u.sdr_gamma);
  return nits_to_pq(n);
}

// ajl_look, appliqué quand look_mix > 0 (//@AJG_LOOK@).
template <class Lut>
AJP_GRADE_FN V3 look_code(V3 e, const Lut& lut, const GradeUniforms& u) {
  if (u.look_mix <= 0.0F) return e;
  if (u.look_pq > 0.5F && u.sdr_out < 0.5F) return look_chain(e, lut, u);
  const V3 nits = look_decode(e, u);
  V3 lin = (u.sdr_out > 0.5F ? nits : mul(bt2020_to_709(), nits)) / u.look_white;
  lin = vmax(lin, 0.0F);
  const float over = gmax(max3(lin), 1.0F);
  const V3 enc = vpow(lin / over, 1.0F / u.look_gamma);
  const V3 graded = vpow(vmax(look_chain(enc, lut, u), 0.0F), u.look_gamma) * over;
  const V3 out_lin = graded * u.look_white;
  return look_encode(u.sdr_out > 0.5F ? out_lin : mul(bt709_to_2020(), out_lin), u);
}

// ajb_black, appliqué quand black_hi > black_lo ; `local_pq` le max local gradé.
AJP_GRADE_FN V3 black_code(V3 e, float local_pq, const GradeUniforms& u) {
  if (u.black_hi <= u.black_lo) return e;
  const float m = look_decode(v3(gmax(max3(e), local_pq)), u).r;
  if (m >= u.black_hi) return e;
  const float gain = smooth(u.black_lo, u.black_hi, m);
  return look_encode(look_decode(e, u) * gain, u);
}

// ajk_kwin, appliqué quand kwin_on > 0,5.
AJP_GRADE_FN V3 kwin_code(V3 e, const GradeUniforms& u) {
  if (u.kwin_on <= 0.5F || u.sdr_out > 0.5F) return e;
  const float k = u.kwin_ref / 203.0F;
  const V3 lms = mul(rgb_to_lms(), (pq_to_nits(e) / 10000.0F) * k);
  const V3 lms_pq = nits_to_pq(lms * 10000.0F);
  const float intensity = 0.5F * (lms_pq.r + lms_pq.g);
  const float luminance = pq_to_nits(v3(intensity)).r;
  float rel = gmax(luminance / u.kwin_ref, 0.0F);
  const float in_range = 10000.0F * k / u.kwin_ref;
  const float out_range = u.kwin_peak / u.kwin_ref;
  const float v = (out_range * (1.0F + in_range) - in_range) / (in_range * in_range);
  rel = rel * (1.0F + rel * v) / (1.0F + rel);
  const float mapped = nits_to_pq(v3(rel * u.kwin_ref)).r;
  const V3 out_lms = pq_to_nits(lms_pq + v3(mapped - intensity));
  const V3 rgb = vclamp(mul(inverse(rgb_to_lms()), out_lms), 0.0F, u.kwin_peak);
  return nits_to_pq(rgb / k);
}

// La sortie d'un pixel du grade : le grade, le look, le vrai noir (max local `black_local`), KWin.
template <class Lut>
AJP_GRADE_FN V3 grade_pixel(V3 code, float skin_sm, float local_raw, float area_pq, float black_local,
                            const GradeUniforms& u, const Lut& lut) {
  V3 e = grade_code(code, skin_sm, local_raw, area_pq, u);
  e = look_code(e, lut, u);
  e = black_code(e, black_local, u);
  return kwin_code(e, u);
}

// Le grade du contrat v2 (chantier 8.6) : une seule lecture de la LUT de la courbe du plan (render::bake,
// indexée par √code PQ sur chaque axe), mêlée à `mix` à celle de la fin d'un fondu (`fade_to`, mix 0 hors
// fondu). Le look SDR et la courbe y sont déjà cuits.
template <class Lut>
AJP_GRADE_FN V3 curve_lut_code(V3 e, const Lut& lut, const Lut& fade_to, float mix) {
  const V3 at = vpow(vclamp(e, 0.0F, 1.0F), 0.5F);
  const V3 c = lut(at);
  if (mix <= 0.0F) return c;
  return vmix(c, fade_to(at), gclamp(mix, 0.0F, 1.0F));
}

// La carte locale (1-local.hook) : le SDR encodé tel que la mesure le prend, et la luminance en PQ.
AJP_GRADE_FN V3 local_sdr_rgb(V3 e, const GradeUniforms& u) {
  if (u.sdr_out > 0.5F) return vclamp(e, 0.0F, 1.0F);
  const V3 n = pq_to_nits(e) / 10000.0F;
  return vpow(vclamp(n * 10000.0F / 203.0F, 0.0F, 1.0F), 1.0F / 2.4F);
}
AJP_GRADE_FN float local_pq_y(V3 e, const GradeUniforms& u) {
  float y;
  if (u.sdr_out > 0.5F)
    y = dot3(vpow(vclamp(e, 0.0F, 1.0F), u.sdr_gamma), V3{0.2126F, 0.7152F, 0.0722F}) * u.peak / 10000.0F;
  else
    y = dot3(pq_to_nits(e) / 10000.0F, V3{0.2627F, 0.6780F, 0.0593F});
  return nits_to_pq(v3(y * 10000.0F)).r;
}
// La valeur d'une lecture : (luminance PQ, bleu, chroma).
AJP_GRADE_FN V3 local_sample(V3 e, const GradeUniforms& u) {
  const V3 c = local_sdr_rgb(e, u);
  return {local_pq_y(e, u), gmax(0.0F, c.b - gmax(c.r, c.g)), max3(c) - min3(c)};
}

// Lecture bilinéaire d'une image à la position normalisée (s, t), bords tenus, comme une texture LINEAR
// CLAMP_TO_EDGE ; fetch(x, y) rend le texel aux indices bornés.
template <class Fetch>
AJP_GRADE_FN V3 bilinear(const Fetch& fetch, int w, int h, float s, float t) {
  const float x = s * static_cast<float>(w) - 0.5F;
  const float y = t * static_cast<float>(h) - 0.5F;
  const float fx = floorf(x);
  const float fy = floorf(y);
  const float ax = x - fx;
  const float ay = y - fy;
  const int x0 = static_cast<int>(fx);
  const int y0 = static_cast<int>(fy);
  const V3 top = vmix(fetch(x0, y0), fetch(x0 + 1, y0), ax);
  const V3 bottom = vmix(fetch(x0, y0 + 1), fetch(x0 + 1, y0 + 1), ax);
  return vmix(top, bottom, ay);
}

// Un texel (lx, ly) de la carte locale de taille lw × lh sur une image w × h : 16 lectures bilinéaires.
template <class Fetch>
AJP_GRADE_FN V3 local_texel(const Fetch& fetch, int w, int h, int lx, int ly, int lw, int lh,
                            const GradeUniforms& u) {
  const float s = (static_cast<float>(lx) + 0.5F) / static_cast<float>(lw);
  const float t = (static_cast<float>(ly) + 0.5F) / static_cast<float>(lh);
  V3 acc = v3(0.0F);
  for (int j = 0; j < 4; j++)
    for (int i = 0; i < 4; i++) {
      const float ox = (static_cast<float>(i) * 2.0F - 3.0F) / static_cast<float>(w);
      const float oy = (static_cast<float>(j) * 2.0F - 3.0F) / static_cast<float>(h);
      acc = acc + local_sample(bilinear(fetch, w, h, s + ox, t + oy), u);
    }
  return acc / 16.0F;
}

// La moyenne 4 × 4 de la carte locale (lw × lh, composante x) sous le texel (qx, qy) de la zone aw × ah,
// indices déjà bornés à la zone : 4 lectures bilinéaires.
template <class Fetch>
AJP_GRADE_FN float area_mean(const Fetch& local, int lw, int lh, int qx, int qy, int aw, int ah) {
  const float s = (static_cast<float>(qx) + 0.5F) / static_cast<float>(aw);
  const float t = (static_cast<float>(qy) + 0.5F) / static_cast<float>(ah);
  float acc = 0.0F;
  for (int j = 0; j < 2; j++)
    for (int k = 0; k < 2; k++)
      acc += bilinear(local, lw, lh, s + (static_cast<float>(k) * 2.0F - 1.0F) / static_cast<float>(lw),
                      t + (static_cast<float>(j) * 2.0F - 1.0F) / static_cast<float>(lh))
                 .r;
  return acc * 0.25F;
}

// Le lissage temporel d'un texel de zone : `kept` la valeur gardée + 1 (0 : jamais écrite), `dt_s` l'écart
// de pts depuis l'image précédente de la même chaîne. Rend la valeur à écrire (l'historique garde elle + 1).
AJP_GRADE_FN float area_smooth(float value, float kept, float dt_s) {
  if (kept > 0.5F && fabsf(kept - 1.0F - value) <= kSmoothJump)
    value = mixf(kept - 1.0F, value, 1.0F - powf(kSmoothBase, kSmoothRate * gmax(dt_s, 0.0F)));
  return value;
}

// Le max local (2b-localmax.hook) du texel (lx, ly) : le pixel le plus clair du bloc 8 × 8, ses indices.
template <class Fetch>
AJP_GRADE_FN void localmax_pick(const Fetch& fetch, int w, int h, int lx, int ly, int lw, int lh, int* at_x,
                                int* at_y) {
  const float s = (static_cast<float>(lx) + 0.5F) / static_cast<float>(lw);
  const float t = (static_cast<float>(ly) + 0.5F) / static_cast<float>(lh);
  const int bx = static_cast<int>(s * static_cast<float>(w)) - 4;
  const int by = static_cast<int>(t * static_cast<float>(h)) - 4;
  float best = -1.0F;
  int ax = bx < 0 ? 0 : (bx > w - 1 ? w - 1 : bx);
  int ay = by < 0 ? 0 : (by > h - 1 ? h - 1 : by);
  for (int j = 0; j < 8; j++)
    for (int i = 0; i < 8; i++) {
      int px = bx + i;
      int py = by + j;
      px = px < 0 ? 0 : (px > w - 1 ? w - 1 : px);
      py = py < 0 ? 0 : (py > h - 1 ? h - 1 : py);
      const float m = max3(fetch(px, py));
      if (m > best) {
        best = m;
        ax = px;
        ay = py;
      }
    }
  *at_x = ax;
  *at_y = ay;
}

}  // namespace ajp::kernels::grade
