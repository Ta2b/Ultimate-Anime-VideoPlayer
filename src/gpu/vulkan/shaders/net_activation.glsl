// Activations des réseaux, dans l'ordre de gpu::ir::Activation et des calculs de l'interpréteur
// (gpu/ir/op_window.cpp, epilogue ; op_elementwise.cpp) : 0 aucune, 1 Relu, 2 LeakyRelu,
// 3 PRelu, 4 Sigmoid, 5 Tanh, 6 Clip. NaN suit fmin et fmax de C pour Clip.
#ifndef AJP_NET_ACTIVATION_GLSL
#define AJP_NET_ACTIVATION_GLSL

const uint kActNone = 0u;
const uint kActRelu = 1u;
const uint kActLeaky = 2u;
const uint kActPrelu = 3u;
const uint kActSigmoid = 4u;
const uint kActTanh = 5u;
const uint kActClip = 6u;

float net_activate(uint act, float v, float alpha, float lo, float hi, float slope) {
  if (act == kActRelu) return v > 0.0 ? v : 0.0;
  if (act == kActLeaky) return v < 0.0 ? v * alpha : v;
  if (act == kActPrelu) return v < 0.0 ? v * slope : v;
  if (act == kActSigmoid) return 1.0 / (1.0 + exp(-v));
  // tanh de GLSL peut déborder par exp : au-delà de 15, tanh vaut 1 en float.
  if (act == kActTanh) return tanh(clamp(v, -15.0, 15.0));
  if (act == kActClip) {
    float low = v > lo || isnan(lo) ? v : lo;
    low = isnan(v) ? lo : low;
    return low < hi || isnan(hi) ? low : hi;
  }
  return v;
}

#endif  // AJP_NET_ACTIVATION_GLSL
