// Colle pleine résolution de RIFE fusionné (kernels/spec/rife_fused.h), commune aux trois noyaux
// rife_fused_entry, rife_fused_glue et rife_fused_finish : portage des noyaux CUDA
// (gpu/cuda/backend/rife_fused_kernels.cu), formules et arrondis FP16 aux mêmes endroits, la grille des
// warps exceptée (pixel_tap). Référence :
// kernels/ref/rife_fused.cpp. S'inclut après tensor.glsl.
#ifndef AJP_RIFE_FUSED_GLSL
#define AJP_RIFE_FUSED_GLSL

// L'arrondi FP16 d'une sortie du graphe (Resize, Mul, Add, Cast, Sigmoid…).
float rounded(float v) {
  return unpackHalf2x16(packHalf2x16(vec2(v, 0.0))).x;
}

// GridSample (coins alignés, bords répétés) : les quatre voisins et les fractions, depuis la position en
// pixels. La grille normalisée du graphe, base + flot × pas ramenée par (g + 1) / 2 × (n - 1), vaut exactement
// x + flot : calculée ainsi, en FP32, elle ne passe ni par les plans 7 à 10 de l'entrée v1 de RIFE, que le
// FP16 arrondit jusqu'à 0,16 px à 1280 de large (A03-20), ni par les sommes FP16 du graphe de DistilDRBA
// (A03-18, A46-04).
struct FusedTap {
  uint x0;
  uint x1;
  uint y0;
  uint y1;
  float wx;
  float wy;
};

FusedTap pixel_tap(uint x, uint y, float flow_x, float flow_y, uint w, uint h) {
  precise float ix = float(x) + flow_x;
  precise float iy = float(y) + flow_y;
  ix = min(max(ix, 0.0), float(w - 1u));
  iy = min(max(iy, 0.0), float(h - 1u));
  FusedTap t;
  t.x0 = uint(ix);
  t.y0 = uint(iy);
  t.x1 = min(t.x0 + 1u, w - 1u);
  t.y1 = min(t.y0 + 1u, h - 1u);
  t.wx = ix - float(t.x0);
  t.wy = iy - float(t.y0);
  return t;
}

float mix4(float a, float b, float d, float e, float wx, float wy) {
  precise float top = a * (1.0 - wx) + b * wx;
  precise float bottom = d * (1.0 - wx) + e * wx;
  precise float v = top * (1.0 - wy) + bottom * wy;
  return v;
}

#endif  // AJP_RIFE_FUSED_GLSL
