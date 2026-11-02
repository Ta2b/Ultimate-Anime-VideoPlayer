// Colle pleine résolution de RIFE fusionné (kernels/spec/rife_fused.h), commune aux trois noyaux
// rife_fused_entry, rife_fused_glue et rife_fused_finish : portage des noyaux CUDA
// (gpu/cuda/backend/rife_fused_kernels.cu), formules et arrondis FP16 aux mêmes endroits. Référence :
// kernels/ref/rife_fused.cpp. S'inclut après tensor.glsl.
#ifndef AJP_RIFE_FUSED_GLSL
#define AJP_RIFE_FUSED_GLSL

// L'arrondi FP16 d'une sortie du graphe (Resize, Mul, Add, Cast, Sigmoid…).
float rounded(float v) {
  return unpackHalf2x16(packHalf2x16(vec2(v, 0.0))).x;
}

// GridSample (coins alignés, bords répétés) : les quatre voisins et les fractions.
struct FusedTap {
  uint x0;
  uint x1;
  uint y0;
  uint y1;
  float wx;
  float wy;
};

FusedTap grid_tap(float gx, float gy, uint w, uint h) {
  precise float ix = (gx + 1.0) * 0.5 * float(w - 1u);
  precise float iy = (gy + 1.0) * 0.5 * float(h - 1u);
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
