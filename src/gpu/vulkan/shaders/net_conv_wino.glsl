// Conv 3 × 3 de pas 1 par Winograd F(2×2, 3×3) (Lavin et Gray, 2015), pour les GPU sans unités
// matricielles (Arc intégré des Core Ultra, Radeon RDNA2, UHD) : 16 produits par tuile de 2 × 2
// pixels et par paire (cin, cout) au lieu de 36, soit 2,25 fois moins. Chaque invocation calcule
// une tuile de sortie pour QUADS × 4 canaux : la tuile d'entrée 4 × 4 de deux canaux à la fois
// (un uint, deux float16), V = Bᵀ d B, puis M += U ⊙ V avec U = G g Gᵀ calculé par l'hôte
// (net_weights.cpp, winograd_weights, rangé [bloc][cin][position][quad] en vec4, lu à l'identique
// par tout le groupe de travail), enfin Y = Aᵀ M A et l'épilogue de net_conv.glsl (biais,
// résiduel ou porte de SPAN, activation, DepthToSpace fondu). Entrée float16 en Nhwc16, groupe
// unique ; les canaux de bourrage ne sont pas lus. L'autotuner compare sa sortie au noyau direct.
// Variante AJP_CONV_PACKED (fp16 de calcul, comme net_conv_f16p) : U en float16, V et les produits
// en f16vec, sommés par paquets de 16 canaux puis ajoutés à M en float.
#include "net_tensor.glsl"
#include "net_activation.glsl"
#ifdef AJP_CONV_PACKED
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#define WINO_D f16vec2
#define WINO_UNPACK unpackFloat2x16
#define WINO_SCALAR float16_t
#else
#define WINO_D vec2
#define WINO_UNPACK unpackHalf2x16
#define WINO_SCALAR float
#endif

layout(local_size_x_id = 31) in;

layout(constant_id = 0) const uint TX = 1;
layout(constant_id = 1) const uint TY = 0;
layout(constant_id = 2) const uint TR = 0;
layout(constant_id = 3) const uint LX = 1;
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
// 16 à 21 (noyau, pas, dilatation) : 3 × 3, pas 1, sans dilatation, vérifiés par l'hôte.
layout(constant_id = 22) const uint PT = 1;
layout(constant_id = 23) const uint PL = 1;
layout(constant_id = 25) const uint ACT = 0;
layout(constant_id = 26) const float ALPHA = 0.0;
layout(constant_id = 27) const float LO = 0.0;
layout(constant_id = 28) const float HI = 0.0;
layout(constant_id = 29) const uint RESIDUAL = 0;
layout(constant_id = 30) const uint QUADS = 1;
// 31 : la taille du groupe de travail ; 32 (PIX de net_conv.glsl) : 1.
layout(constant_id = 33) const uint SPACE = 0;      // DepthToSpace fondu : blocs de SPACE × SPACE
layout(constant_id = 34) const uint SPACE_CRD = 0;  // 1 : ordre CRD, 0 : DCR

const uint BLOCKS = (COUT + 4u * QUADS - 1u) / (4u * QUADS);
const uint TILES_W = (WOUT + 1u) / 2u;

AJP_NET_BUFFER(0, src, readonly)
layout(std430, binding = 0) readonly buffer SourceU { uint v[]; } src_u;  // deux float16 par uint
#ifdef AJP_CONV_PACKED
layout(std430, binding = 1) readonly buffer Weights { f16vec4 w[]; } weights;
#else
layout(std430, binding = 1) readonly buffer Weights { vec4 w[]; } weights;
#endif
layout(std430, binding = 2) readonly buffer Bias { float v[]; } bias;
layout(std430, binding = 3) readonly buffer Slope { float v[]; } slope;
AJP_NET_BUFFER(4, res, readonly)
AJP_NET_BUFFER(5, dst, writeonly)

void main() {
  const TDesc xd = TDesc(N, CIN, HIN, WIN, LX, CXP);
  const uint sb = SPACE > 1u ? SPACE : 1u;
  const TDesc yd = TDesc(N, COUT / (sb * sb), HOUT * sb, WOUT * sb, LY, CYP);
  const TDesc rd = TDesc(N, COUT, HOUT, WOUT, LR, CRP);
  const uint tx = gl_GlobalInvocationID.x;
  const uint oy0 = gl_WorkGroupID.y * 2u;
  const uint n = gl_WorkGroupID.z / BLOCKS;
  const uint blk = gl_WorkGroupID.z % BLOCKS;
  if (tx >= TILES_W) return;
  const uint ox0 = tx * 2u;

  // La tuile d'entrée 4 × 4 : premier uint de chaque ligne et de chaque colonne (CXP / 2 uint par
  // pixel), et lesquelles tombent dans l'image ; hors d'elle, des zéros.
  uint row_at[4];
  uint col_at[4];
  bool row_in[4];
  bool col_in[4];
  for (uint r = 0u; r < 4u; ++r) {
    const int iy = int(oy0 + r) - int(PT);
    const int ix = int(ox0 + r) - int(PL);
    row_in[r] = iy >= 0 && iy < int(HIN);
    col_in[r] = ix >= 0 && ix < int(WIN);
    row_at[r] = row_in[r] ? (n * HIN + uint(iy)) * WIN * (CXP / 2u) : 0u;
    col_at[r] = col_in[r] ? uint(ix) * (CXP / 2u) : 0u;
  }

  vec4 m[16 * QUADS];
  for (uint a = 0u; a < 16u * QUADS; ++a) m[a] = vec4(0.0);
#ifdef AJP_CONV_PACKED
  f16vec4 part[16 * QUADS];
  for (uint a = 0u; a < 16u * QUADS; ++a) part[a] = f16vec4(0.0);
#endif
  const uint pairs = CIN / 2u;
  for (uint cp = 0u; cp <= pairs; ++cp) {
    // Deux canaux par uint ; le dernier, seul, quand cin est impair.
    const bool single = cp == pairs;
    if (single && CIN % 2u == 0u) break;
    const uint ci = cp * 2u;
    WINO_D d[16];
    for (uint r = 0u; r < 4u; ++r) {
      for (uint c = 0u; c < 4u; ++c) {
        WINO_D v = WINO_D(0.0);
        if (row_in[r] && col_in[c]) {
          if (single) {
            const uint iy = oy0 + r - PT;
            const uint ix = ox0 + c - PL;
            v.x = WINO_SCALAR(AJP_LOAD(src, TX, t_index(xd, n, ci, iy, ix)));
          } else {
            v = WINO_UNPACK(src_u.v[row_at[r] + col_at[c] + cp]);
          }
        }
        d[r * 4u + c] = v;
      }
    }
    // V = Bᵀ d B : les lignes, puis les colonnes ; Bᵀ = [1 0 -1 0 ; 0 1 1 0 ; 0 -1 1 0 ; 0 1 0 -1].
    for (uint c = 0u; c < 4u; ++c) {
      const WINO_D d0 = d[c];
      const WINO_D d1 = d[4u + c];
      const WINO_D d2 = d[8u + c];
      const WINO_D d3 = d[12u + c];
      d[c] = d0 - d2;
      d[4u + c] = d1 + d2;
      d[8u + c] = d2 - d1;
      d[12u + c] = d1 - d3;
    }
    for (uint r = 0u; r < 4u; ++r) {
      const WINO_D t0 = d[r * 4u];
      const WINO_D t1 = d[r * 4u + 1u];
      const WINO_D t2 = d[r * 4u + 2u];
      const WINO_D t3 = d[r * 4u + 3u];
      d[r * 4u] = t0 - t2;
      d[r * 4u + 1u] = t1 + t2;
      d[r * 4u + 2u] = t2 - t1;
      d[r * 4u + 3u] = t1 - t3;
    }
    const uint w0 = (blk * CIN + ci) * 16u * QUADS;
#ifdef AJP_CONV_PACKED
    for (uint p = 0u; p < 16u; ++p) {
      for (uint q = 0u; q < QUADS; ++q) {
        part[p * QUADS + q] += d[p].x * weights.w[w0 + p * QUADS + q];
        if (!single) part[p * QUADS + q] += d[p].y * weights.w[w0 + (16u + p) * QUADS + q];
      }
    }
    // Tous les 16 canaux, et au dernier : les sommes float16 passent en float.
    if (cp % 8u == 7u || cp + 1u >= pairs + (CIN % 2u)) {
      for (uint a = 0u; a < 16u * QUADS; ++a) {
        m[a] += vec4(part[a]);
        part[a] = f16vec4(0.0);
      }
    }
#else
    for (uint p = 0u; p < 16u; ++p) {
      for (uint q = 0u; q < QUADS; ++q) {
        m[p * QUADS + q] += d[p].x * weights.w[w0 + p * QUADS + q];
        if (!single) m[p * QUADS + q] += d[p].y * weights.w[w0 + (16u + p) * QUADS + q];
      }
    }
#endif
  }

  for (uint q = 0u; q < QUADS; ++q) {
    // Y = Aᵀ M A, Aᵀ = [1 1 1 0 ; 0 1 -1 -1] : les lignes, puis les colonnes.
    vec4 s[8];
    for (uint c = 0u; c < 4u; ++c) {
      const vec4 m0 = m[c * QUADS + q];
      const vec4 m1 = m[(4u + c) * QUADS + q];
      const vec4 m2 = m[(8u + c) * QUADS + q];
      const vec4 m3 = m[(12u + c) * QUADS + q];
      s[c] = m0 + m1 + m2;
      s[4u + c] = m1 - m2 - m3;
    }
    vec4 y[4];
    for (uint u = 0u; u < 2u; ++u) {
      y[u * 2u] = s[u * 4u] + s[u * 4u + 1u] + s[u * 4u + 2u];
      y[u * 2u + 1u] = s[u * 4u + 1u] - s[u * 4u + 2u] - s[u * 4u + 3u];
    }
    for (uint u = 0u; u < 2u; ++u) {
      const uint oy = oy0 + u;
      if (oy >= HOUT) break;
      for (uint w = 0u; w < 2u; ++w) {
        const uint ox = ox0 + w;
        if (ox >= WOUT) break;
        for (uint k = 0u; k < 4u; ++k) {
          const uint co = blk * 4u * QUADS + 4u * q + k;
          if (co >= COUT) continue;
          float v = y[u * 2u + w][k] + bias.v[co];
          if (RESIDUAL == 1u) v += AJP_LOAD(res, TR, t_index(rd, n, co, oy, ox));
          v = net_activate(ACT, v, ALPHA, LO, HI, slope.v[co]);
          // Porte de SPAN : sortie de la forme de x, lue au même élément.
          if (RESIDUAL == 2u) v *= AJP_LOAD(src, TX, t_index(xd, n, co, oy, ox)) + AJP_LOAD(res, TR, t_index(rd, n, co, oy, ox));
          AJP_STORE(dst, TY, t_index_space(yd, sb, SPACE_CRD == 1u, n, co, oy, ox), v)
        }
      }
    }
  }
}
