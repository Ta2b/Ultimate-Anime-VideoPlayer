// Conv 2D par matrices coopératives FP8 (VK_KHR_cooperative_matrix, tuiles E4M3 × E4M3 → float
// 16 × 16 × 16 des RDNA 4), variante de net_conv_coopmat.glsl pour une Conv dont l'entrée est
// arrondie à la grille E4M3 (ConvAttrs::in_quant = IN_Q) et dont les poids, rangés par l'hôte en
// E4M3 (net_weights.cpp, matrix_weights_fp8), valent q × échelle de leur canal de sortie.
// Même découpage que la variante float16 : M les pixels de sortie, N les canaux de sortie arrondis
// à 16, K = kh × kw × cinp ; une prise (ky, kx) et KC canaux par étape ; A de l'étape (KC canaux de
// BM pixels, nuls hors de l'image et au-delà de cin) copiée en mémoire partagée, en double tampon,
// une barrière par étape ; B (KC × Np octets) lue dans le tampon ou copiée aussi (SHARED_B).
// A : les canaux float16 de l'image divisés par IN_Q puis encodés en E4M3 à la copie (e4m3_bits,
// le même arrondi que gpu/ir/fp8.h) ; produits en E4M3, sommes en float. L'épilogue multiplie la
// somme par IN_Q × échelle du canal (rangée à la suite des biais : bias.v[NP + canal]), puis suit
// l'ordre de l'IR : biais, arrondi de sortie (OUT_Q), résiduel, activation, porte de SPAN. Entrée
// float16 en Nhwc16 exigée. Référence : gpu/ir/op_window.cpp (eval_conv).
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_float_e4m3 : require
#include "net_tensor.glsl"
#include "net_activation.glsl"

layout(local_size_x_id = 31) in;

layout(constant_id = 0) const uint TX = 0;
layout(constant_id = 1) const uint TY = 0;
layout(constant_id = 2) const uint TR = 0;
layout(constant_id = 3) const uint LX = 0;
layout(constant_id = 4) const uint LY = 0;
layout(constant_id = 5) const uint LR = 0;
layout(constant_id = 6) const uint CXP = 16;
layout(constant_id = 7) const uint CYP = 16;
layout(constant_id = 8) const uint CRP = 16;
layout(constant_id = 9) const uint N = 1;
layout(constant_id = 10) const uint CIN = 1;
layout(constant_id = 11) const uint HIN = 1;
layout(constant_id = 12) const uint WIN = 1;
layout(constant_id = 13) const uint COUT = 1;
layout(constant_id = 14) const uint HOUT = 1;
layout(constant_id = 15) const uint WOUT = 1;
layout(constant_id = 16) const uint KH = 1;
layout(constant_id = 17) const uint KW = 1;
layout(constant_id = 18) const uint SH = 1;
layout(constant_id = 19) const uint SW = 1;
layout(constant_id = 20) const uint DH = 1;
layout(constant_id = 21) const uint DW = 1;
layout(constant_id = 22) const uint PT = 0;
layout(constant_id = 23) const uint PL = 0;
layout(constant_id = 24) const uint GROUPS = 1;
layout(constant_id = 25) const uint ACT = 0;
layout(constant_id = 26) const float ALPHA = 0.0;
layout(constant_id = 27) const float LO = 0.0;
layout(constant_id = 28) const float HI = 0.0;
layout(constant_id = 29) const uint RESIDUAL = 0;
layout(constant_id = 30) const uint TM = 2;
layout(constant_id = 32) const uint SGS = 32;
layout(constant_id = 33) const uint KC = 16;
layout(constant_id = 34) const uint SPACE = 0;      // DepthToSpace fondu : blocs de SPACE × SPACE
layout(constant_id = 35) const uint SPACE_CRD = 0;  // 1 : ordre CRD, 0 : DCR
layout(constant_id = 36) const uint SHARED_B = 0;   // 1 : B de l'étape en mémoire partagée, comme A
layout(constant_id = 37) const float IN_Q = 1.0;    // échelle de l'entrée (> 0, toujours)
layout(constant_id = 38) const float OUT_Q = 0.0;   // > 0 : résultat arrondi à la grille E4M3

const uint NT = (COUT + 15u) / 16u;
const uint NP = NT * 16u;
const uint SUBGROUPS = gl_WorkGroupSize.x / SGS;
const uint BM = SUBGROUPS * TM * 16u;
const uint QA = KC / 16u;                       // uvec4 de 16 canaux E4M3 par pixel et par étape
const uint AS = QA + (QA >= 2u ? 1u : 0u);      // pas d'une ligne de A, en uvec4 (+1 dès KC = 32)
const uint ABUF = BM * AS;                      // un tampon de A
const uint CSTEPS = CXP / KC;
const uint STEPS = KH * KW * CSTEPS;
const uint QB = KC * NP / 16u;                  // uvec4 de B par étape
const uint BBUF = SHARED_B == 1u ? QB : 1u;     // un tampon de B

layout(std430, binding = 0) readonly buffer Source { uvec4 v[]; } src4;
layout(std430, binding = 1) readonly buffer Weights { uvec4 v[]; } weights4;
layout(std430, binding = 2) readonly buffer Bias { float v[]; } bias;
layout(std430, binding = 3) readonly buffer Slope { float v[]; } slope;
AJP_NET_BUFFER(4, res, readonly)
AJP_NET_BUFFER(5, dst, writeonly)

shared uvec4 tile_a[2u * ABUF];
shared uvec4 tile_b[2u * BBUF];
shared float tile_c[SUBGROUPS * 256u];

// Octet E4M3 (variante « fn », saturant à 448, arrondi au plus proche pair) de v ; le même calcul
// que float_to_e4m3 de gpu/ir/fp8.h.
uint e4m3_bits(float v) {
  if (isnan(v)) return 0x7Fu;
  uint sign = (floatBitsToUint(v) >> 24u) & 0x80u;
  float a = abs(v);
  if (a >= 448.0) return sign | 0x7Eu;
  if (a == 0.0) return sign;
  int e;
  frexp(a, e);
  float step = ldexp(1.0, max(e - 1, -6) - 3);
  float m = roundEven(a / step) * step;
  if (m >= 448.0) return sign | 0x7Eu;
  if (m == 0.0) return sign;
  int e2;
  float f2 = frexp(m, e2);
  if (e2 - 1 < -6) return sign | uint(roundEven(ldexp(m, 9)));
  return sign | (uint(e2 + 6) << 3u) | uint(roundEven((f2 * 2.0 - 1.0) * 8.0));
}

// Les demi-flottants d'un uvec4 (huit canaux) divisés par IN_Q, ceux d'indice ≥ valid nuls, en
// octets E4M3 : deux mots (quatre octets chacun).
uvec2 quantize8(uvec4 v, uint valid) {
  uvec2 bytes = uvec2(0u);
  for (uint k = 0u; k < 4u; ++k) {
    vec2 h = unpackHalf2x16(v[k]);
    uint lo = 2u * k < valid ? e4m3_bits(h.x / IN_Q) : 0u;
    uint hi = 2u * k + 1u < valid ? e4m3_bits(h.y / IN_Q) : 0u;
    bytes[k / 2u] |= (lo | (hi << 8u)) << (16u * (k % 2u));
  }
  return bytes;
}

// x au pixel (y, x) et au canal c, pour la porte de SPAN (sortie de la forme de x), non arrondi.
float source_at(uint n, uint c, uint y, uint x) {
  uint e = ((n * HIN + y) * WIN + x) * CXP + c;
  vec2 h = unpackHalf2x16(src4.v[e / 8u][(e % 8u) / 2u]);
  return (e & 1u) == 0u ? h.x : h.y;
}

// A de l'étape s dans le tampon buf : une invocation par uvec4 de 16 canaux d'un pixel, invocations
// consécutives sur les uvec4 consécutifs d'un pixel, puis du pixel suivant.
void stage_a(uint buf, uint n, uint first_pixel, uint s) {
  uint c0 = (s % CSTEPS) * KC;
  uint tap = s / CSTEPS;
  uint ky = tap / KW;
  uint kx = tap % KW;
  for (uint i = gl_LocalInvocationIndex; i < BM * QA; i += gl_WorkGroupSize.x) {
    uint r = i / QA;
    uint c = c0 + (i % QA) * 16u;
    uvec4 out16 = uvec4(0u);
    uint p = first_pixel + r;
    if (p < HOUT * WOUT && c < CIN) {
      int iy = int((p / WOUT) * SH + ky * DH) - int(PT);
      int ix = int((p % WOUT) * SW + kx * DW) - int(PL);
      if (iy >= 0 && iy < int(HIN) && ix >= 0 && ix < int(WIN)) {
        uint base = (((n * HIN + uint(iy)) * WIN + uint(ix)) * CXP + c) / 8u;
        uvec2 lo = quantize8(src4.v[base], CIN - c);
        uvec2 hi = CIN - c > 8u ? quantize8(src4.v[base + 1u], CIN - c - 8u) : uvec2(0u);
        out16 = uvec4(lo, hi);
      }
    }
    tile_a[buf * ABUF + r * AS + i % QA] = out16;
  }
}

// B de l'étape s dans le tampon buf : ses KC lignes de NP octets se suivent dans le tampon des
// poids, copiées d'un bloc.
void stage_b(uint buf, uint s) {
  uint k0 = (s / CSTEPS) * CXP + (s % CSTEPS) * KC;
  for (uint i = gl_LocalInvocationIndex; i < QB; i += gl_WorkGroupSize.x) {
    tile_b[buf * BBUF + i] = weights4.v[k0 * (NP / 16u) + i];
  }
}

void main() {
  // La sortie stockée : celle du DepthToSpace fondu quand SPACE > 1.
  const uint sb = SPACE > 1u ? SPACE : 1u;
  const TDesc yd = TDesc(N, COUT / (sb * sb), HOUT * sb, WOUT * sb, LY, CYP);
  const TDesc rd = TDesc(N, COUT, HOUT, WOUT, LR, CRP);
  uint n = gl_WorkGroupID.z;
  uint first_pixel = gl_WorkGroupID.x * BM;
  uint sg = gl_SubgroupID;

  coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc[TM * NT];
  for (uint i = 0u; i < TM * NT; ++i) {
    acc[i] = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
  }
  stage_a(0u, n, first_pixel, 0u);
  if (SHARED_B == 1u) stage_b(0u, 0u);
  barrier();
  for (uint s = 0u; s < STEPS; ++s) {
    uint buf = s & 1u;
    // L'autre tampon a été lu à l'étape s − 1, avant la barrière qui l'a close.
    if (s + 1u < STEPS) {
      stage_a(buf ^ 1u, n, first_pixel, s + 1u);
      if (SHARED_B == 1u) stage_b(buf ^ 1u, s + 1u);
    }
    uint k0 = (s / CSTEPS) * CXP + (s % CSTEPS) * KC;
    for (uint c16 = 0u; c16 < KC; c16 += 16u) {
      coopmat<floate4m3_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> a[TM];
      for (uint t = 0u; t < TM; ++t) {
        coopMatLoad(a[t], tile_a, buf * ABUF + (sg * TM + t) * 16u * AS + c16 / 16u, AS,
                    gl_CooperativeMatrixLayoutRowMajor);
      }
      uint k = k0 + c16;
      for (uint j = 0u; j < NT; ++j) {
        coopmat<floate4m3_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> b;
        if (SHARED_B == 1u) {
          coopMatLoad(b, tile_b, buf * BBUF + c16 * (NP / 16u) + j, NP / 16u, gl_CooperativeMatrixLayoutRowMajor);
        } else {
          coopMatLoad(b, weights4.v, k * (NP / 16u) + j, NP / 16u, gl_CooperativeMatrixLayoutRowMajor);
        }
        for (uint t = 0u; t < TM; ++t) acc[t * NT + j] = coopMatMulAdd(a[t], b, acc[t * NT + j]);
      }
    }
    barrier();
  }
  for (uint t = 0u; t < TM; ++t) {
    for (uint j = 0u; j < NT; ++j) {
      coopMatStore(acc[t * NT + j], tile_c, sg * 256u, 16u, gl_CooperativeMatrixLayoutRowMajor);
      subgroupMemoryBarrierShared();
      subgroupBarrier();
      for (uint e = gl_SubgroupInvocationID; e < 256u; e += gl_SubgroupSize) {
        uint p = first_pixel + (sg * TM + t) * 16u + e / 16u;
        uint co = j * 16u + e % 16u;
        if (p < HOUT * WOUT && co < COUT) {
          uint oy = p / WOUT;
          uint ox = p % WOUT;
          float v = tile_c[sg * 256u + e] * bias.v[NP + co] + bias.v[co];
          if (OUT_Q > 0.0) v = round_e4m3(v / OUT_Q) * OUT_Q;
          if (RESIDUAL == 1u) v += AJP_LOAD(res, TR, t_index(rd, n, co, oy, ox));
          v = net_activate(ACT, v, ALPHA, LO, HI, ACT == kActPrelu ? slope.v[co] : 0.0);
          if (RESIDUAL == 2u) v *= source_at(n, co, oy, ox) + AJP_LOAD(res, TR, t_index(rd, n, co, oy, ox));
          AJP_STORE(dst, TY, t_index_space(yd, sb, SPACE_CRD == 1u, n, co, oy, ox), v)
        }
      }
      subgroupBarrier();
    }
  }
}
