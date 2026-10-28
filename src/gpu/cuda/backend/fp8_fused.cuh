// Prototype (non branché au CMake du lecteur) : noyau élémentaire fusionné de la queue d'un bloc
// SPAN en FP8, « Add(saut) + Tanh(attention) + Mul + requantification E4M3 » en une seule passe
// mémoire. Il remplace, côté TensorRT, les noyaux Myelin séparés TanhAddMulMulCast + MulCast.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ajp::gpu::cuda {

// Échelles de déquantification (valeur réelle = octet FP8 × échelle). out_inv est l'inverse de
// l'échelle de sortie : l'octet écrit est sat_e4m3(valeur × out_inv).
struct Fp8SpanTailScales {
  float c3 = 1.0f;
  float skip = 1.0f;
  float out_inv = 1.0f;
};

// out = sat_e4m3( (c3 × s_c3 + skip × s_skip) × tanh(att) × out_inv ), élément par élément.
//   c3, skip, out : octets E4M3 ; att : conv d'attention en FP16 (avant le Tanh) ;
//   n multiple de 16 (16 octets FP8 par thread, deux chargements de 16 octets pour att).
// tanh_fast : tanh.approx.f32 (erreur relative ~2^-11, sans effet devant l'arrondi E4M3).
cudaError_t launch_fp8_span_tail(const uint8_t* c3, const __half* att, const uint8_t* skip, uint8_t* out,
                                 size_t n, const Fp8SpanTailScales& scales, bool tanh_fast,
                                 cudaStream_t stream);

// Variante où att est lui aussi en E4M3 (échelle att_scale) : 4 octets lus + 1 écrit par élément
// au lieu de 5.
cudaError_t launch_fp8_span_tail_att8(const uint8_t* c3, const uint8_t* att, const uint8_t* skip,
                                      uint8_t* out, size_t n, const Fp8SpanTailScales& scales,
                                      float att_scale, cudaStream_t stream);

}  // namespace ajp::gpu::cuda
