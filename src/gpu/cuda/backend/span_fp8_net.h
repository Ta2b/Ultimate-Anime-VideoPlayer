// Le SPAN entier en FP8 (noyaux CUTLASS, docs/span-fp8-cutlass.md), vu de l'extérieur : aucun en-tête CUTLASS
// ici, ils restent dans span_fp8_net.cu. Dégroupage, conv_1 (FP16), huit blocs SPAB en FP8 E4M3,
// concaténation des quatre tenseurs, conv_cat, to_pixel (FP16) et DepthToSpace, lancés l'un après l'autre sur
// le flux donné. Entrée 1×3×H×W et sortie 1×3×2H×2W en FP16 NCHW, comme le SPAN en ONNX.
//
// Cette interface n'emploie ni std::expected ni <format> : nvcc (EDG) ne les lit pas, et le .cu qui
// l'implémente est compilé par lui. span_fp8_engine.cpp la convertit en Refusal.
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>

namespace ajp::gpu::cuda {

// Une convolution vue sans copie : poids FP16 (motifs binaires) rangés [sortie][ligne][colonne][entrée],
// biais f32.
struct SpanFp8ConvView {
  const std::uint16_t* weights = nullptr;
  const float* bias = nullptr;
};

// Le contenu d'un gpu/span_fp8/package.h, sans ses vecteurs. Les pointeurs ne servent que pendant create().
struct SpanFp8Weights {
  static constexpr int kBlocks = 8;
  static constexpr int kScales = 1 + 3 * kBlocks;
  const float* scales = nullptr;  // kScales échelles E4M3 d'activation
  SpanFp8ConvView first;
  SpanFp8ConvView conv[kBlocks][3];
  SpanFp8ConvView attention[kBlocks];
  SpanFp8ConvView concat;
  SpanFp8ConvView pixel;
};

enum class SpanFp8Status : std::uint8_t { Ok, OutOfMemory, Failed };

class SpanFp8Net {
 public:
  // Vrai pour sm 8.9 (Ada) seulement : les noyaux emploient les mma FP8 de cette architecture et ne sont
  // compilés que pour elle.
  static bool supported_device(int major, int minor);

  // Envoie les poids sur l'appareil courant (FP8 pour les blocs) et prépare les lancements. height et width
  // sont déjà jugés par span_fp8::size_refusal. Nul si l'envoi échoue ; status et error disent pourquoi.
  static std::unique_ptr<SpanFp8Net> create(const SpanFp8Weights& weights, int height, int width,
                                            SpanFp8Status& status, std::string& error);
  ~SpanFp8Net();
  SpanFp8Net(const SpanFp8Net&) = delete;
  SpanFp8Net& operator=(const SpanFp8Net&) = delete;

  // Octets de travail que launch() demande (tampons intermédiaires), à fournir 256 octets alignés.
  std::int64_t workspace_bytes() const;

  // Pose la zone de travail : les paramètres des lancements portent ses adresses. À refaire quand la zone
  // change ; sans appel aux fonctions CUDA de flux, donc sans effet sur une capture de graphe en cours.
  void bind(void* workspace);

  // Lance les noyaux sur stream. input : 3×H×W FP16, output : 3×2H×2W FP16 (adresses d'appareil). Faux si un
  // lancement échoue (cudaGetLastError), avec le motif dans error.
  bool launch(cudaStream_t stream, const void* input, void* output, std::string& error);

 private:
  struct Impl;
  explicit SpanFp8Net(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace ajp::gpu::cuda
