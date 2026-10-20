// Conv 2D par matrices coopératives (VK_KHR_cooperative_matrix), en produit implicite : M = les
// pixels de sortie, N = les canaux de sortie arrondis à 16, K = kh × kw × cinp (cinp : canaux
// d'entrée rangés en Nhwc16). Chaque sous-groupe calcule TM tuiles de 16 pixels pour toutes les
// tuiles de 16 canaux ; un groupe de travail de WG invocations en compte WG / SGS. Le produit
// avance par étapes : une prise (ky, kx) et un paquet de KC canaux (KC divise cinp). La matrice A
// de l'étape (KC canaux de BM pixels, nuls hors de l'image et au-delà de cin) est copiée en
// mémoire partagée, en double tampon : la copie de l'étape suivante part avant les produits de
// l'étape courante, une seule barrière par étape. Les lignes de A sont espacées de KC / 8 + 1
// uvec4 dès KC = 32, contre les conflits de bancs. B (KC × Np, poids rangés par net_weights.cpp,
// matrix_weights) est lue dans le tampon. Produits en float16, sommes en float. L'épilogue
// (biais, résiduel, activation) passe tuile par tuile par la mémoire partagée, puis écrit la
// sortie dans sa disposition et son type. Entrée float16 en Nhwc16 exigée (l'hôte ne propose
// cette variante qu'alors). Référence : gpu/ir/op_window.cpp (eval_conv).
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
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

const uint NT = (COUT + 15u) / 16u;
const uint NP = NT * 16u;
const uint SUBGROUPS = gl_WorkGroupSize.x / SGS;
const uint BM = SUBGROUPS * TM * 16u;
const uint QA = KC / 8u;                        // uvec4 de canaux par pixel et par étape
const uint AS = QA + (QA / 4u + 1u) / 2u;      // pas d'une ligne de A, en uvec4 (+1 dès KC = 32)
const uint ABUF = BM * AS;                      // un tampon de A
const uint CSTEPS = CXP / KC;
const uint STEPS = KH * KW * CSTEPS;

layout(std430, binding = 0) readonly buffer Source { uvec4 v[]; } src4;
layout(std430, binding = 1) readonly buffer Weights { float16_t w[]; } weights;
layout(std430, binding = 2) readonly buffer Bias { float v[]; } bias;
layout(std430, binding = 3) readonly buffer Slope { float v[]; } slope;
AJP_NET_BUFFER(4, res, readonly)
AJP_NET_BUFFER(5, dst, writeonly)

shared uvec4 tile_a[2u * ABUF];
shared float tile_c[SUBGROUPS * 256u];

// Les demi-flottants d'indice ≥ valid d'un paquet de 8 (un uvec4) mis à zéro.
uvec4 keep_halves(uvec4 v, uint valid) {
  for (uint k = 0u; k < 4u; ++k) {
    if (2u * k >= valid) {
      v[k] = 0u;
    } else if (2u * k + 1u >= valid) {
      v[k] &= 0xFFFFu;
    }
  }
  return v;
}

// A de l'étape s dans le tampon buf : invocations consécutives sur des uvec4 consécutifs d'un
// pixel, puis du pixel suivant (lectures contiguës quand KC = cinp).
void stage_a(uint buf, uint n, uint first_pixel, uint s) {
  uint c0 = (s % CSTEPS) * KC;
  uint tap = s / CSTEPS;
  uint ky = tap / KW;
  uint kx = tap % KW;
  for (uint i = gl_LocalInvocationIndex; i < BM * QA; i += gl_WorkGroupSize.x) {
    uint r = i / QA;
    uint c = c0 + (i % QA) * 8u;
    uvec4 v = uvec4(0u);
    uint p = first_pixel + r;
    if (p < HOUT * WOUT && c < CIN) {
      int iy = int((p / WOUT) * SH + ky * DH) - int(PT);
      int ix = int((p % WOUT) * SW + kx * DW) - int(PL);
      if (iy >= 0 && iy < int(HIN) && ix >= 0 && ix < int(WIN)) {
        v = src4.v[(((n * HIN + uint(iy)) * WIN + uint(ix)) * CXP + c) / 8u];
        if (c + 8u > CIN) v = keep_halves(v, CIN - c);
      }
    }
    tile_a[buf * ABUF + r * AS + i % QA] = v;
  }
}

void main() {
  const TDesc yd = TDesc(N, COUT, HOUT, WOUT, LY, CYP);
  const TDesc rd = TDesc(N, COUT, HOUT, WOUT, LR, CRP);
  uint n = gl_WorkGroupID.z;
  uint first_pixel = gl_WorkGroupID.x * BM;
  uint sg = gl_SubgroupID;

  coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc[TM * NT];
  for (uint i = 0u; i < TM * NT; ++i) {
    acc[i] = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
  }
  stage_a(0u, n, first_pixel, 0u);
  barrier();
  for (uint s = 0u; s < STEPS; ++s) {
    uint buf = s & 1u;
    // L'autre tampon a été lu à l'étape s − 1, avant la barrière qui l'a close.
    if (s + 1u < STEPS) stage_a(buf ^ 1u, n, first_pixel, s + 1u);
    uint k0 = (s / CSTEPS) * CXP + (s % CSTEPS) * KC;
    for (uint c16 = 0u; c16 < KC; c16 += 16u) {
      coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> a[TM];
      for (uint t = 0u; t < TM; ++t) {
        coopMatLoad(a[t], tile_a, buf * ABUF + (sg * TM + t) * 16u * AS + c16 / 8u, AS,
                    gl_CooperativeMatrixLayoutRowMajor);
      }
      uint k = k0 + c16;
      for (uint j = 0u; j < NT; ++j) {
        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> b;
        coopMatLoad(b, weights.w, k * NP + j * 16u, NP, gl_CooperativeMatrixLayoutRowMajor);
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
          float v = tile_c[sg * 256u + e] + bias.v[co];
          if (RESIDUAL == 1u) v += AJP_LOAD(res, TR, t_index(rd, n, co, oy, ox));
          v = net_activate(ACT, v, ALPHA, LO, HI, slope.v[co]);
          AJP_STORE(dst, TY, t_index(yd, n, co, oy, ox), v)
        }
      }
      subgroupBarrier();
    }
  }
}
