// Activations des réseaux, dans l'ordre de gpu::ir::Activation et des calculs de l'interpréteur
// (gpu/ir/op_window.cpp, epilogue ; op_elementwise.cpp) : 0 aucune, 1 Relu, 2 LeakyRelu,
// 3 PRelu, 4 Sigmoid, 5 Tanh, 6 Clip, puis, pour net_unary seulement, 7 Reciprocal, 8 FakeQuantE4M3
// (arrondi à la grille FP8 E4M3 de l'échelle alpha, gpu/ir/fp8.h) et 9 FakeQuantInt8 (grille INT8,
// gpu/ir/int8.h). NaN suit fmin et fmax de C pour Clip.
#ifndef KIRARI_NET_ACTIVATION_GLSL
#define KIRARI_NET_ACTIVATION_GLSL

const uint kActNone = 0u;
const uint kActRelu = 1u;
const uint kActLeaky = 2u;
const uint kActPrelu = 3u;
const uint kActSigmoid = 4u;
const uint kActTanh = 5u;
const uint kActClip = 6u;
const uint kActReciprocal = 7u;
const uint kActFakeE4m3 = 8u;
const uint kActFakeInt8 = 9u;

// x arrondi à la grille FP8 E4M3 (au plus proche pair, saturant à 448), comme round_to_e4m3 de
// gpu/ir/fp8.h : pas 2^-9 sous 2^-6, sinon 2^(e - 3) pour x dans [2^e, 2^(e+1)[.
float round_e4m3(float x) {
  if (isnan(x)) return x;
  float a = abs(x);
  if (a >= 448.0) return x < 0.0 ? -448.0 : 448.0;
  if (a == 0.0) return x;
  int e;
  frexp(a, e);  // a = f × 2^e, f dans [0,5 ; 1[
  float step = ldexp(1.0, max(e - 1, -6) - 3);
  float r = roundEven(a / step) * step;
  r = min(r, 448.0);
  return x < 0.0 ? -r : r;
}

// x arrondi à l'entier le plus proche (pair au milieu), saturé à [-128, 127], comme round_to_int8 de
// gpu/ir/int8.h ; NaN donne 0 (clamp de GLSL ne dit rien de NaN).
float round_int8(float x) {
  if (isnan(x)) return 0.0;
  return clamp(roundEven(x), -128.0, 127.0);
}

// L'arrondi des Conv quantifiées (ConvAttrs::in_quant et out_quant, gpu::ir::quant_round) : q > 0, la
// grille E4M3 d'échelle q ; q < 0, la grille INT8 d'échelle -q (ConvAttrs::quant_grid, signe posé par
// net_lower_conv.cpp) ; 0, rien.
float net_quant(float v, float q) {
  if (q > 0.0) return round_e4m3(v / q) * q;
  if (q < 0.0) return round_int8(v / -q) * -q;
  return v;
}

float net_activate(uint act, float v, float alpha, float lo, float hi, float slope) {
  if (act == kActRelu) return v > 0.0 ? v : 0.0;
  if (act == kActLeaky) return v < 0.0 ? v * alpha : v;
  if (act == kActPrelu) return v < 0.0 ? v * slope : v;
  if (act == kActSigmoid) return 1.0 / (1.0 + exp(-v));
  if (act == kActReciprocal) return 1.0 / v;
  if (act == kActFakeE4m3) return round_e4m3(v / alpha) * alpha;
  if (act == kActFakeInt8) return round_int8(v / alpha) * alpha;
  // tanh de GLSL peut déborder par exp : au-delà de 15, tanh vaut 1 en float.
  if (act == kActTanh) return tanh(clamp(v, -15.0, 15.0));
  if (act == kActClip) {
    float low = v > lo || isnan(lo) ? v : lo;
    low = isnan(v) ? lo : low;
    return low < hi || isnan(hi) ? low : hi;
  }
  return v;
}

#endif  // KIRARI_NET_ACTIVATION_GLSL
