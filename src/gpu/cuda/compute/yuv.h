// Conversions YUV ↔ RGB planaire de kernels/spec/color.h sur l'appareil.
//
// Ingestion : l'image décodée (NVDEC, mémoire linéaire à pas de ligne : NV12, P010, P016 et les
// formats planaires) devient du RGB planaire FP16 ou FP32, avec sa matrice, sa plage et son siting ;
// la chroma 4:2:0 brute passe en horizontal puis en vertical, puis la matrice s'applique
// (kernels::ref::yuv_to_rgb). Avec rife_pad, l'image est placée au centre de son remplissage
// mod-64 en noir studio, comme kernels::ref::rife_pad, sans copie : les lectures hors de la
// source rendent les codes du noir.
//
// Sortie : RGB planaire → YUV quantifié avec l'arrondi exact de quantize() (P010 compris), Cb et
// Cr normalisés passant en horizontal puis en vertical (kernels::ref::rgb_to_yuv), puis la
// fenêtre demandée seulement (sortie de RIFE : kernels::ref::rife_finish).
//
// Un plan prépare ses tables de poids et ses intermédiaires dans le pool une fois ; run()
// n'alloue rien et ne copie rien côté hôte.
#pragma once

#include <cuda_runtime_api.h>

#include "gpu/cuda/compute/image.h"
#include "gpu/cuda/compute/memory.h"
#include "gpu/cuda/compute/weights.h"
#include "kernels/spec/rife.h"
#include "kernels/spec/warp.h"

namespace ajp::gpu::cuda {

struct IngestSpec {
  kernels::Format format = kernels::Format::Nv12;
  int width = 0;  // image décodée
  int height = 0;
  ColorSpec color;
  kernels::Filter chroma = kernels::kChainChromaFilter;
  bool rife_pad = false;  // sortie de kernels::rife_geometry(width, height) remplie
};

// La conversion d'entrée de RIFE (spec/rife.h) : remplissage, BT.709 quelle que soit la source,
// chroma bilinéaire en siting gauche.
IngestSpec rife_ingest_spec(kernels::Format format, int width, int height, kernels::Range range);

class IngestPlan {
 public:
  // Refus : dimensions hors contrat (paires en 4:2:0), filtre Area (il ne fait que réduire,
  // compute_weights le refuse), pool insuffisant.
  static Result<IngestPlan> create(DevicePool& pool, const IngestSpec& spec, cudaStream_t stream);

  int output_width() const { return out_width_; }
  int output_height() const { return out_height_; }

  // src : image du format et de la taille de la spécification ; rgb : 3 plans de
  // output_width() × output_height().
  Result<> run(const YuvPlanes& src, const Tensor& rgb, cudaStream_t stream) const;

 private:
  IngestSpec spec_;
  int out_width_ = 0;
  int out_height_ = 0;
  int left_ = 0;
  int top_ = 0;
  dev::YuvToRgb matrix_;
  DeviceWeights wx_;
  DeviceWeights wy_;
  DeviceBuffer tmp_;
};

struct OutputSpec {
  kernels::Format format = kernels::Format::P010;
  int width = 0;  // tenseur RGB
  int height = 0;
  ColorSpec color;  // la chaîne passe kernels::kOutputSiting
  kernels::Filter chroma = kernels::kChainChromaFilter;
  // Fenêtre écrite ; vide : le tenseur entier. En 4:2:0, coin et dimensions pairs.
  kernels::Window window;
};

// La conversion de sortie de RIFE et son recadrage (kernels::ref::rife_finish).
OutputSpec rife_output_spec(kernels::Format format, kernels::Range range,
                            const kernels::RifeGeometry& geometry);

class OutputPlan {
 public:
  static Result<OutputPlan> create(DevicePool& pool, const OutputSpec& spec, cudaStream_t stream);

  const kernels::Window& window() const { return spec_.window; }

  // rgb : 3 plans de width × height ; dst : image du format, de la taille de la fenêtre.
  Result<> run(const Tensor& rgb, const YuvPlanes& dst, cudaStream_t stream) const;

 private:
  OutputSpec spec_;
  dev::RgbToYuv matrix_;
  DeviceWeights wx_;
  DeviceWeights wy_;
  DeviceBuffer tmp_;   // Cb et Cr normalisés, width × height
  DeviceBuffer rows_;  // après la passe horizontale, width / 2 × height
};

// RGB → RGBA16 UNORM entrelacé, alpha opaque (kernels::ref::to_rgba16), dans une mémoire
// linéaire de même taille (pas multiple de 8, adresse alignée sur 8).
Result<> write_rgba16(const Tensor& rgb, const Plane& dst, cudaStream_t stream);

}  // namespace ajp::gpu::cuda
