// Le SPAN entier en FP16, par les mêmes convolutions CUTLASS que le réseau FP8 (span_fp8_net.h) mais sans
// aucune quantification : la référence de la dérivation native du paquet « .span8 »
// (docs/span-fp8-cutlass.md). Elle sert deux fois : à relever le maximum de |valeur| de chacun des 25
// tenseurs que le réseau FP8 quantifie (la calibration, sur des images réelles : mêmes définitions que
// tools/span/derive_span_fp8.py, qui les relève en FP32 sur ONNX Runtime), et à produire la sortie contre
// laquelle la sortie FP8 est jugée.
//
// Elle ne remplace pas le FP16 de TensorRT dans le lecteur : c'est un outil de dérivation, lent et gourmand
// (1 Ko de zone de travail par pixel de caractéristiques), qui calcule le même réseau en FP16 avec
// accumulation FP32 ; le dernier chiffre de son arrondi n'est pas celui de TensorRT. Sans std::expected ni
// <format> : nvcc ne les lit pas.
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>

#include "gpu/cuda/backend/span_fp8_net.h"

namespace ajp::gpu::cuda {

class SpanFp8Reference {
 public:
  static constexpr int kPeaks = SpanFp8Weights::kScales;

  // Envoie les poids (FP16, les échelles de weights sont ignorées) et réserve la zone de travail pour des
  // images height × width, déjà jugées par span_fp8::size_refusal. Nul si l'envoi échoue.
  static std::unique_ptr<SpanFp8Reference> create(const SpanFp8Weights& weights, int height, int width,
                                                  SpanFp8Status& status, std::string& error);
  ~SpanFp8Reference();
  SpanFp8Reference(const SpanFp8Reference&) = delete;
  SpanFp8Reference& operator=(const SpanFp8Reference&) = delete;

  // Passe une image : input 3×H×W FP16 NCHW, output 3×2H×2W FP16 (adresses d'appareil). track : le maximum
  // de |valeur| de chaque tenseur est ajouté aux maxima gardés (peaks()). Synchrone sur stream. Faux si un
  // lancement échoue, avec le motif dans error.
  bool run(cudaStream_t stream, const void* input, void* output, bool track, std::string& error);

  // Les maxima de |valeur| gardés depuis la création, dans l'ordre des échelles du paquet : [0] sortie de
  // conv_1 ; bloc b : [1+3b] premier Relu, [2+3b] second Relu, [3+3b] sortie du bloc.
  bool peaks(float (&out)[kPeaks], std::string& error);

 private:
  struct Impl;
  explicit SpanFp8Reference(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace ajp::gpu::cuda
