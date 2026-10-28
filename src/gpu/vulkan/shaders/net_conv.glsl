// Conv 2D directe, générique : groupes, dilatation, pas, marges, biais, résiduel et activation
// fondus (gpu::ir::Epilogue : biais, puis résiduel, puis activation ; RESIDUAL 2, la porte de
// SPAN, multiplie la sortie activée par x + r au lieu d'ajouter r avant ; SPACE > 1 écrit chaque
// valeur à sa place après un DepthToSpace). Chaque invocation calcule
// PIX pixels voisins d'une ligne de sortie pour QUADS × 4 canaux d'un même groupe ; un groupe de
// travail couvre WG × PIX pixels d'une ligne. PIX > 1 lit chaque poids une fois pour PIX pixels
// (et les colonnes d'entrée communes à deux pixels voisins une fois du cache), au prix de
// registres : l'autotuner tranche. Les poids sont rangés par l'hôte en [groupe × bloc][kh][kw][cin][4 × QUADS]
// (net_weights.cpp), lus en vec4 identiques pour tout le groupe de travail.
// Variante « _f16 » : tenseurs float16 permis, calcul en float. Variante « _f16p » (AJP_CONV_PACKED,
// fp16 de calcul) : poids float16, produits en f16vec4 sommés par paquets de 16 canaux, puis
// ajoutés en float. Une entrée float16 en Nhwc16 sans groupes (VEC) se lit par uvec4, huit
// canaux d'un pixel par lecture. Référence : gpu/ir/op_window.cpp (eval_conv).
#include "net_tensor.glsl"
#include "net_activation.glsl"
#ifdef AJP_CONV_PACKED
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif

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
layout(constant_id = 30) const uint QUADS = 1;
layout(constant_id = 32) const uint PIX = 1;  // 31 : la taille du groupe de travail
layout(constant_id = 33) const uint SPACE = 0;      // DepthToSpace fondu : blocs de SPACE × SPACE
layout(constant_id = 34) const uint SPACE_CRD = 0;  // 1 : ordre CRD, 0 : DCR

const uint CING = CIN / GROUPS;
const uint COUTG = COUT / GROUPS;
const uint BLOCKS = (COUTG + 4u * QUADS - 1u) / (4u * QUADS);

AJP_NET_BUFFER(0, src, readonly)
#ifdef AJP_NET_F16
layout(std430, binding = 0) readonly buffer SourceV { uvec4 v[]; } src_v;
const bool VEC = TX == 1u && LX == 1u && GROUPS == 1u;
#endif
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
  // La sortie stockée : celle du DepthToSpace fondu quand SPACE > 1.
  const uint sb = SPACE > 1u ? SPACE : 1u;
  const TDesc yd = TDesc(N, COUT / (sb * sb), HOUT * sb, WOUT * sb, LY, CYP);
  const TDesc rd = TDesc(N, COUT, HOUT, WOUT, LR, CRP);
  uint ox0 = gl_GlobalInvocationID.x * PIX;
  uint oy = gl_WorkGroupID.y;
  uint n = gl_WorkGroupID.z / (GROUPS * BLOCKS);
  uint gb = gl_WorkGroupID.z % (GROUPS * BLOCKS);
  if (ox0 >= WOUT) return;
  uint g = gb / BLOCKS;
  uint first_out = (gb % BLOCKS) * 4u * QUADS;

  // acc[p * QUADS + q] : pixel ox0 + p, canaux first_out + 4q à 4q + 3.
  vec4 acc[PIX * QUADS];
  for (uint a = 0u; a < PIX * QUADS; ++a) acc[a] = vec4(0.0);
  for (uint ky = 0u; ky < KH; ++ky) {
    int iy = int(oy * SH + ky * DH) - int(PT);
    if (iy < 0 || iy >= int(HIN)) continue;
    for (uint kx = 0u; kx < KW; ++kx) {
      // Colonne d'entrée de chaque pixel ; hors de l'image (ou pixel au-delà de la ligne), rien.
      int ix[PIX];
      bool inside[PIX];
      bool any = false;
      for (uint p = 0u; p < PIX; ++p) {
        ix[p] = int((ox0 + p) * SW + kx * DW) - int(PL);
        inside[p] = ix[p] >= 0 && ix[p] < int(WIN) && ox0 + p < WOUT;
        any = any || inside[p];
      }
      if (!any) continue;
      uint row = ((gb * KH + ky) * KW + kx) * CING * QUADS;
#ifdef AJP_NET_F16
      if (VEC) {
        // Huit canaux par uvec4 ; les canaux de bourrage (au-delà de cin) ne sont pas lus.
        uint base[PIX];
        for (uint p = 0u; p < PIX; ++p)
          base[p] = inside[p] ? (((n * HIN + uint(iy)) * WIN + uint(ix[p])) * CXP) / 8u : 0u;
        uint full = CING / 8u;
        for (uint c8 = 0u; c8 < full; ++c8) {
          uvec4 packed[PIX];
          for (uint p = 0u; p < PIX; ++p) packed[p] = inside[p] ? src_v.v[base[p] + c8] : uvec4(0u);
#ifdef AJP_CONV_PACKED
          f16vec4 part[PIX * QUADS];
          for (uint a = 0u; a < PIX * QUADS; ++a) part[a] = f16vec4(0.0);
#endif
          for (uint h = 0u; h < 4u; ++h) {
            uint ci = c8 * 8u + 2u * h;
            for (uint q = 0u; q < QUADS; ++q) {
#ifdef AJP_CONV_PACKED
              f16vec4 w0 = weights.w[row + ci * QUADS + q];
              f16vec4 w1 = weights.w[row + (ci + 1u) * QUADS + q];
              for (uint p = 0u; p < PIX; ++p) {
                f16vec2 xv = unpackFloat2x16(packed[p][h]);
                part[p * QUADS + q] += xv.x * w0;
                part[p * QUADS + q] += xv.y * w1;
              }
#else
              vec4 w0 = weights.w[row + ci * QUADS + q];
              vec4 w1 = weights.w[row + (ci + 1u) * QUADS + q];
              for (uint p = 0u; p < PIX; ++p) {
                vec2 xv = unpackHalf2x16(packed[p][h]);
                acc[p * QUADS + q] += xv.x * w0;
                acc[p * QUADS + q] += xv.y * w1;
              }
#endif
            }
          }
#ifdef AJP_CONV_PACKED
          for (uint a = 0u; a < PIX * QUADS; ++a) acc[a] += vec4(part[a]);
#endif
        }
        for (uint ci = full * 8u; ci < CING; ++ci) {
          float xv[PIX];
          for (uint p = 0u; p < PIX; ++p)
            xv[p] = inside[p] ? AJP_LOAD(src, TX, t_index(xd, n, ci, uint(iy), uint(ix[p]))) : 0.0;
          for (uint q = 0u; q < QUADS; ++q) {
            vec4 wv = vec4(weights.w[row + ci * QUADS + q]);
            for (uint p = 0u; p < PIX; ++p) acc[p * QUADS + q] += xv[p] * wv;
          }
        }
        continue;
      }
#endif
#ifdef AJP_CONV_PACKED
      for (uint c0 = 0u; c0 < CING; c0 += 16u) {
        f16vec4 part[PIX * QUADS];
        for (uint a = 0u; a < PIX * QUADS; ++a) part[a] = f16vec4(0.0);
        uint c1 = min(c0 + 16u, CING);
        for (uint ci = c0; ci < c1; ++ci) {
          float16_t xv[PIX];
          for (uint p = 0u; p < PIX; ++p) {
            xv[p] = inside[p] ? float16_t(AJP_LOAD(src, TX, t_index(xd, n, g * CING + ci, uint(iy), uint(ix[p]))))
                              : float16_t(0.0);
          }
          for (uint q = 0u; q < QUADS; ++q) {
            f16vec4 wv = weights.w[row + ci * QUADS + q];
            for (uint p = 0u; p < PIX; ++p) part[p * QUADS + q] += xv[p] * wv;
          }
        }
        for (uint a = 0u; a < PIX * QUADS; ++a) acc[a] += vec4(part[a]);
      }
#else
      for (uint ci = 0u; ci < CING; ++ci) {
        float xv[PIX];
        for (uint p = 0u; p < PIX; ++p)
          xv[p] = inside[p] ? AJP_LOAD(src, TX, t_index(xd, n, g * CING + ci, uint(iy), uint(ix[p]))) : 0.0;
        for (uint q = 0u; q < QUADS; ++q) {
          vec4 wv = weights.w[row + ci * QUADS + q];
          for (uint p = 0u; p < PIX; ++p) acc[p * QUADS + q] += xv[p] * wv;
        }
      }
#endif
    }
  }

  for (uint p = 0u; p < PIX; ++p) {
    uint ox = ox0 + p;
    if (ox >= WOUT) break;
    for (uint q = 0u; q < QUADS; ++q) {
      for (uint k = 0u; k < 4u; ++k) {
        uint o = first_out + 4u * q + k;
        if (o >= COUTG) continue;
        uint co = g * COUTG + o;
        float v = acc[p * QUADS + q][k] + bias.v[co];
        if (RESIDUAL == 1u) v += AJP_LOAD(res, TR, t_index(rd, n, co, oy, ox));
        v = net_activate(ACT, v, ALPHA, LO, HI, ACT == kActPrelu ? slope.v[co] : 0.0);
        // Porte de SPAN : sortie de la forme de x, lue au même élément.
        if (RESIDUAL == 2u) v *= AJP_LOAD(src, TX, t_index(xd, n, co, oy, ox)) + AJP_LOAD(res, TR, t_index(rd, n, co, oy, ox));
        AJP_STORE(dst, TY, t_index_space(yd, sb, SPACE_CRD == 1u, n, co, oy, ox), v)
      }
    }
  }
}
