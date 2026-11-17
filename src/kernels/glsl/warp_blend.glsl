// Warp arrière par flot et mélange de deux images à l'instant t : base de Fast et de
// l'ordonnancement S3. À t = 1/2 exactement, le mouvement et le masque estimés par RIFE tels
// quels ; sinon chaque point suit la ligne droite du mouvement et la loi de mélange ajustée.
// Garder les poses (keep_poses) : l'instant de chaque pixel se règle d'après le mouvement au
// milieu de la paire précédente (kernels/spec/poses.h), puis suit le même chemin.
// Contrat : kernels/spec/warp.h ; référence : kernels/ref/warp.cpp (warp_blend, warp_blend_at,
// warp_blend_poses).
// Repris de animejanai-inference-rtx@118c272 src/kernels.cu:963-1013 (sample_plane,
// sample_window, k_warp_blend) et 1051-1096 (k_warp_blend_mid).
// Le corps de warp_blend.comp (sortie dans un tenseur) et de warp_blend_ring.comp (sortie dans le slot du
// ring, AJP_WARP_RING), inclus après tensor.glsl.
#ifndef AJP_WARP_BLEND_GLSL
#define AJP_WARP_BLEND_GLSL

#include "ajp_kernels_spec.glsl"
#include "common.glsl"

layout(local_size_x = 16, local_size_y = 16) in;

// 1 : les images a et b lues en Catmull-Rom (spec/warp.h : RifeWarp), le flot et le masque restant bilinéaires.
layout(constant_id = 0) const uint CUBIC = 0;

layout(std430, binding = 0) restrict readonly buffer ImageA { AJP_TENSOR v[]; } a;
layout(std430, binding = 1) restrict readonly buffer ImageB { AJP_TENSOR v[]; } b;
layout(std430, binding = 2) restrict readonly buffer Flow { AJP_TENSOR v[]; } flow;  // f0x, f0y, f1x, f1y
layout(std430, binding = 3) restrict readonly buffer Mask { AJP_TENSOR v[]; } mask;  // poids de a
#ifdef AJP_WARP_RING
// Le slot du ring par sa vue R16G16B16A16_UNORM, comme ring_store (écrit après les tampons).
#define AJP_WARP_PREVIOUS 4
#define AJP_WARP_PARAMS 5
#define AJP_WARP_PAN 6
layout(binding = 7, rgba16) uniform writeonly image2D slot;
#else
layout(std430, binding = 4) writeonly buffer Blend { AJP_TENSOR v[]; } dst;
#define AJP_WARP_PREVIOUS 5
#define AJP_WARP_PARAMS 6
#define AJP_WARP_PAN 7
#endif
// Mouvement au milieu de la paire précédente, mêmes plans et même fenêtre ; lu seulement avec
// keep_poses et has_previous (sinon le flot y est lié à sa place).
layout(std430, binding = AJP_WARP_PREVIOUS) readonly buffer Previous { AJP_TENSOR v[]; } previous;
// L'instant, mot de paramètre réécrit d'une exécution à l'autre sans réenregistrer.
layout(std430, binding = AJP_WARP_PARAMS) readonly buffer Params { float t; } params;
// Mouvement d'ensemble de la paire (pan_global.comp, spec/warp_pan.h) : dx, dy, 1 si sûre ; lu seulement avec
// pan_lock (sinon l'instant y est lié à sa place).
layout(std430, binding = AJP_WARP_PAN) readonly buffer Pan { float v[]; } pan;

layout(push_constant) uniform Push {
  uint width;  // images a, b et sortie
  uint height;
  uint flow_width;  // plans du mouvement, souvent remplis
  uint flow_height;
  uint area_x;  // fenêtre de l'image dans les plans du mouvement
  uint area_y;
  uint area_width;
  uint area_height;
  uint keep_poses;    // 1 : garder les poses
  uint has_previous;  // 0 : paire précédente tenue, sans mouvement
  uint pan_lock;      // 1 : mouvement d'ensemble (spec/warp_pan.h), hors garder les poses
} pc;

// Les quatre voisins d'une lecture bilinéaire et ses fractions (spec/warp.h) : coordonnées
// bornées à la fenêtre (NaN donne 0), x0 = tronque(x), x1 = min(x0 + 1, w − 1).
struct Taps {
  uint i00;
  uint i10;
  uint i01;
  uint i11;
  float fx;
  float fy;
};

Taps taps_in(uint stride, uint x0w, uint y0w, uint w, uint h, float x, float y) {
  x = bounded(x, 0.0, float(w - 1u));
  y = bounded(y, 0.0, float(h - 1u));
  uint x0 = uint(x);
  uint y0 = uint(y);
  uint x1 = min(x0 + 1u, w - 1u);
  uint y1 = min(y0 + 1u, h - 1u);
  uint r0 = (y0w + y0) * stride + x0w;
  uint r1 = (y0w + y1) * stride + x0w;
  return Taps(r0 + x0, r0 + x1, r1 + x0, r1 + x1, x - float(x0), y - float(y0));
}

float bilinear(float p00, float p10, float p01, float p11, Taps t) {
  precise float top = p00 * (1.0 - t.fx) + p10 * t.fx;
  precise float bottom = p01 * (1.0 - t.fx) + p11 * t.fx;
  precise float value = top * (1.0 - t.fy) + bottom * t.fy;
  return value;
}

// Les voisins d'un point (u, v) de la fenêtre du mouvement, relatifs à son coin : calculés une fois pour
// tous les plans lus au même point (comme sample_n sous CUDA ; même résultat que plan par plan).
Taps motion_taps(float u, float v) {
  return taps_in(pc.flow_width, pc.area_x, pc.area_y, pc.area_width, pc.area_height, u, v);
}

// Les quatre plans du mouvement (f0x, f0y, f1x, f1y) aux voisins t.
vec4 flow_at(Taps t) {
  uint plane = pc.flow_width * pc.flow_height;
  vec4 at;
  for (uint c = 0u; c < 4u; ++c) {
    uint base = c * plane;
    at[c] = bilinear(float(flow.v[base + t.i00]), float(flow.v[base + t.i10]), float(flow.v[base + t.i01]),
                     float(flow.v[base + t.i11]), t);
  }
  return at;
}

// Le masque (poids de a) aux voisins t.
float mask_at(Taps t) {
  return bilinear(float(mask.v[t.i00]), float(mask.v[t.i10]), float(mask.v[t.i01]), float(mask.v[t.i11]), t);
}

// Le point (x, y) du plan c de a ou de b, sans borne.
float image_texel(bool from_a, uint index) {
  return from_a ? float(a.v[index]) : float(b.v[index]);
}

// Catmull-Rom (ref::catmull_rom_at) : position bornée à l'image, points répétés au bord, sortie bornée aux quatre points centraux (C50-08).
vec4 catmull_rom(float t) {
  float t2 = t * t;
  float t3 = t2 * t;
  return vec4(-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0, -1.5 * t3 + 2.0 * t2 + 0.5 * t,
              0.5 * t3 - 0.5 * t2);
}

float image_cubic(bool from_a, uint c, float x, float y) {
  x = bounded(x, 0.0, float(pc.width - 1u));
  y = bounded(y, 0.0, float(pc.height - 1u));
  int x0 = int(x);
  int y0 = int(y);
  vec4 wx = catmull_rom(x - float(x0));
  vec4 wy = catmull_rom(y - float(y0));
  uint base = c * pc.width * pc.height;
  precise float sum = 0.0;
  float lo = 3.4e38;
  float hi = -3.4e38;
  for (int j = 0; j < 4; ++j) {
    uint row_at = base + uint(clamp(y0 - 1 + j, 0, int(pc.height) - 1)) * pc.width;
    precise float row = 0.0;
    for (int i = 0; i < 4; ++i) {
      float v = image_texel(from_a, row_at + uint(clamp(x0 - 1 + i, 0, int(pc.width) - 1)));
      row += v * wx[i];
      if ((i == 1 || i == 2) && (j == 1 || j == 2)) {
        lo = min(lo, v);
        hi = max(hi, v);
      }
    }
    sum += row * wy[j];
  }
  return min(max(sum, lo), hi);
}

float image_at(bool from_a, uint c, float x, float y) {
  if (CUBIC == 1u) return image_cubic(from_a, c, x, y);
  Taps t = taps_in(pc.width, 0u, 0u, pc.width, pc.height, x, y);
  uint base = c * pc.width * pc.height;
  if (from_a) return bilinear(float(a.v[base + t.i00]), float(a.v[base + t.i10]), float(a.v[base + t.i01]),
                              float(a.v[base + t.i11]), t);
  return bilinear(float(b.v[base + t.i00]), float(b.v[base + t.i10]), float(b.v[base + t.i01]),
                  float(b.v[base + t.i11]), t);
}

// Les plans first à first + count − 1 (count ≤ 4) du mouvement de la paire précédente au point (u, v), lus
// comme flow_at.
vec4 previous_at(uint first, uint count, float u, float v) {
  Taps t = motion_taps(u, v);
  uint plane = pc.flow_width * pc.flow_height;
  vec4 at = vec4(0.0);
  for (uint k = 0u; k < count; ++k) {
    uint base = (first + k) * plane;
    at[k] = bilinear(float(previous.v[base + t.i00]), float(previous.v[base + t.i10]),
                     float(previous.v[base + t.i01]), float(previous.v[base + t.i11]), t);
  }
  return at;
}

// L'instant réglé du pixel (u, v) pour garder les poses (keeping_poses de ref/warp.cpp) : le
// déplacement de la paire, celui de la paire précédente remonté jusqu'au même point, et
// pose_time (spec/poses.h). at_uv : le mouvement lu en (u, v), le même à chaque itération.
float pose_time_at(float u, float v, float t, vec4 at_uv) {
  float tp = t;
  for (int k = 0; k < AJP_POSE_ITERATIONS; ++k) {
    precise float u2 = u - ((1.0 - tp) * at_uv.x + tp * at_uv.z);
    precise float v2 = v - ((1.0 - tp) * at_uv.y + tp * at_uv.w);
    vec4 at2 = flow_at(motion_taps(u2, v2));
    precise float dx = at2.z - at2.x;
    precise float dy = at2.w - at2.y;
    float before = 0.0;
    if (pc.has_previous != 0u) {
      precise float ax = u - tp * dx;
      precise float ay = v - tp * dy;
      float qx = ax;
      float qy = ay;
      for (int step = 0; step < 2; ++step) {
        vec4 back = previous_at(2u, 2u, qx, qy);
        precise float nx = ax - back.x;
        precise float ny = ay - back.y;
        qx = nx;
        qy = ny;
      }
      vec4 at_q = previous_at(0u, 4u, qx, qy);
      precise float px = at_q.z - at_q.x;
      precise float py = at_q.w - at_q.y;
      before = sqrt(px * px + py * py);
    }
    precise float now = sqrt(dx * dx + dy * dy);
    precise float r = (before + AJP_POSE_EPSILON) / (before + now + 2.0 * AJP_POSE_EPSILON);
    precise float timed = t < 0.5 ? 2.0 * t * r : 1.0 - 2.0 * (1.0 - t) * r;
    tp = bounded(timed, AJP_POSE_TIME_LIMIT, 1.0 - AJP_POSE_TIME_LIMIT);
  }
  return tp;
}

// Déplacements vers a (xy) et vers b (zw), et poids de a, au point (u, v) du mouvement.
struct Displacement {
  vec4 to;
  float weight;
};

// Le mouvement d'ensemble : le flot lu f remplacé par la translation de la paire au milieu, en pixels de la
// fenêtre, là où il est petit et proche d'elle (spec/warp_pan.h).
// pan_lock hors garder les poses : le fondu sous un demi-pixel ; le flot n'est remplacé que si l'estimation
// est sûre (pan.v[2]).
bool pan_on() {
  return pc.pan_lock != 0u && pc.keep_poses == 0u;
}

vec4 pan_flow(vec4 f) {
  float sx = float(pc.width) / float(pc.area_width);
  float sy = float(pc.height) / float(pc.area_height);
  if (pan.v[2] < 0.5) return f;
  vec4 g = vec4(-0.5 * pan.v[0] / sx, -0.5 * pan.v[1] / sy, 0.5 * pan.v[0] / sx, 0.5 * pan.v[1] / sy);
  vec4 af = abs(f);
  vec4 ae = abs(f - g);
  float m = max(max(af.x, af.y), max(af.z, af.w));
  float e = max(max(ae.x, ae.y), max(ae.z, ae.w));
  return (m < AJP_PAN_FLOW_LIMIT && e < AJP_PAN_AGREEMENT) ? g : f;
}

float slow_fade(float d) {
  return smoothstep(AJP_SLOW_FADE_LO, AJP_SLOW_FADE_HI, d);
}

Displacement estimated(vec4 at_uv, Taps uv, float t) {
  float weight = mask_at(uv);
  if (pan_on()) {  // sous un demi-pixel, le masque glisse vers le fondu (ln((1 − t) / t), nul à t = 1/2)
    float mm = bounded(weight, AJP_MASK_EPSILON, 1.0 - AJP_MASK_EPSILON);
    float w = slow_fade(length(at_uv.zw - at_uv.xy));
    precise float z = w * log(mm / (1.0 - mm)) + (1.0 - w) * log((1.0 - t) / t);
    weight = 1.0 / (1.0 + exp(-z));
  }
  return Displacement(at_uv, weight);
}

// Un autre instant depuis le mouvement du milieu : où le point vu en t était au milieu, son
// mouvement de là, et la loi logit(m) = A logit(mm) + B ln((1 − t) / t) + C. at_uv : le mouvement lu
// en (u, v).
Displacement from_middle(float u, float v, float t, vec4 at_uv) {
  precise float u2 = u - ((1.0 - t) * at_uv.x + t * at_uv.z);
  precise float v2 = v - ((1.0 - t) * at_uv.y + t * at_uv.w);
  Taps at = motion_taps(u2, v2);
  vec4 at2 = flow_at(at);
  bool on = pan_on();
  if (on) at2 = pan_flow(at2);
  precise float tx = at2.z - at2.x;
  precise float ty = at2.w - at2.y;
  float mm = bounded(mask_at(at), AJP_MASK_EPSILON, 1.0 - AJP_MASK_EPSILON);
  precise float z =
      AJP_BLEND_LAW_A * log(mm / (1.0 - mm)) + AJP_BLEND_LAW_B * log((1.0 - t) / t) + AJP_BLEND_LAW_C;
  if (on) {
    float w = slow_fade(sqrt(tx * tx + ty * ty));
    z = w * z + (1.0 - w) * log((1.0 - t) / t);
  }
  precise float weight = 1.0 / (1.0 + exp(-z));
  return Displacement(vec4(-t * tx, -t * ty, (1.0 - t) * tx, (1.0 - t) * ty), weight);
}

#if defined(AJP_WARP_RING) && defined(AJP_TENSOR_F16)
// v arrondi au float16 le plus proche, moitié vers le pair : la valeur que l'écriture d'un float16_t dans le
// tenseur garde (conversion au plus proche des pilotes visés). Écrit en float : le stockage 16 bits seul n'a pas
// de float16_t en calcul, et unpackHalf2x16(packHalf2x16(v)) se simplifie en v chez certains compilateurs
// (llvmpipe). Multiplier et diviser par une puissance de deux sont exacts. Au-delà de 65504, l'infini ou NaN :
// rendus tels quels, unorm16 les borne de même.
float half_rounded(float v) {
  float a = abs(v);
  if (!(a < 65504.0)) return v;
  int e = a < 6.103515625e-05 ? -14 : ((floatBitsToInt(a) >> 23) - 127);
  float scale = ldexp(1.0, 10 - e);
  return roundEven(v * scale) / scale;
}
#endif

void main() {
  uint x = gl_GlobalInvocationID.x;
  uint y = gl_GlobalInvocationID.y;
  if (x >= pc.width || y >= pc.height) return;
  // Où le pixel tombe dans la grille du mouvement, centres alignés.
  precise float sx = float(pc.width) / float(pc.area_width);
  precise float sy = float(pc.height) / float(pc.area_height);
  precise float u = (float(x) + 0.5) / sx - 0.5;
  precise float v = (float(y) + 0.5) / sy - 0.5;
  // Le mouvement en (u, v), lu une fois pour les poses, le mélange et l'instant du milieu.
  Taps uv = motion_taps(u, v);
  vec4 at_uv = flow_at(uv);
  Displacement d;
  if (pc.keep_poses != 0u) {
    d = from_middle(u, v, pose_time_at(u, v, params.t, at_uv), at_uv);
  } else {
    if (pan_on()) at_uv = pan_flow(at_uv);
    d = params.t == 0.5 ? estimated(at_uv, uv, params.t) : from_middle(u, v, params.t, at_uv);
  }
  precise float ax = float(x) + sx * d.to.x;
  precise float ay = float(y) + sy * d.to.y;
  precise float bx = float(x) + sx * d.to.z;
  precise float by = float(y) + sy * d.to.w;
  vec3 rgb;
  for (uint c = 0u; c < 3u; ++c) {
    float va = image_at(true, c, ax, ay);
    float vb = image_at(false, c, bx, by);
    precise float mixed = va * d.weight + vb * (1.0 - d.weight);
    rgb[c] = mixed;
  }
#ifdef AJP_WARP_RING
  // Le code que write_ring écrirait du tenseur de sortie, au bit près : la valeur arrondie à la précision
  // du tenseur (float16 : half_rounded), puis unorm16 (ring_store.comp), écrite code / 65535 que la vue UNORM
  // rend exactement.
  vec4 codes = vec4(1.0);
  for (uint c = 0u; c < 3u; ++c) {
#ifdef AJP_TENSOR_F16
    float stored = half_rounded(rgb[c]);
#else
    float stored = rgb[c];
#endif
    codes[c] = float(unorm16(stored)) / AJP_UNORM16_MAX;
  }
  imageStore(slot, ivec2(x, y), codes);
#else
  uint plane = pc.width * pc.height;
  for (uint c = 0u; c < 3u; ++c) dst.v[c * plane + y * pc.width + x] = AJP_TENSOR(rgb[c]);
#endif
}

#endif  // AJP_WARP_BLEND_GLSL
