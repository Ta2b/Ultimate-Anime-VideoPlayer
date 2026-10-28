// Fonctions de kernels/spec.h partagées par les noyaux, écrites dans l'ordre des formules de
// la spécification. « precise » interdit la contraction en FMA : les arrondis restent ceux de
// la référence CPU de kernels/ref/. S'inclut après ajp_kernels_spec.glsl.
#ifndef AJP_COMMON_GLSL
#define AJP_COMMON_GLSL

// mirror_index (spec/resample.h) : miroir sans répétition du bord, puis clamp.
int mirror_index(int i, int n) {
  i = i < 0 ? -i - 1 : i;
  i = i >= n ? 2 * n - 1 - i : i;
  return i < 0 ? 0 : (i >= n ? n - 1 : i);
}

// v borné à [lo, hi] ; NaN donne lo, comme fmaxf (clamp() est indéfini pour NaN).
float bounded(float v, float lo, float hi) {
  return v > lo ? (v < hi ? v : hi) : lo;
}

// quantize (spec/color.h) : code = clamp(roundEven(raw / qdiv), 0, qmax) × qdiv, q = (qdiv, qmax).
// NaN et valeurs négatives donnent 0, au-delà de qmax donne qmax.
uint quantize(float raw, vec2 q) {
  precise float x = raw / q.x;
  precise float code = x >= q.y ? q.y : (x > 0.0 ? roundEven(x) : 0.0);
  return uint(code * q.x);
}

// unorm16 (spec/color.h) : tronque(clamp(v, 0, 1) × 65535 + 0,5) ; NaN donne 0.
uint unorm16(float v) {
  float unit = bounded(v, 0.0, 1.0);
  precise float biased = unit * AJP_UNORM16_MAX + 0.5;
  return uint(biased);
}

#endif  // AJP_COMMON_GLSL
