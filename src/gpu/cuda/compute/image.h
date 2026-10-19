// Ce que les plans hôtes reçoivent : tenseurs planaires et images YUV de l'appareil, et les
// vérifications faites avant chaque lancement. Une entrée hors contrat est refusée, jamais
// corrigée : ShapeInvalid pour une taille, InputMismatch pour une mémoire mal décrite (adresse
// nulle ou mal alignée, pas de ligne trop court).
#pragma once

#include <array>
#include <cstddef>
#include <string_view>

#include "gpu/cuda/compute/kernels/views.h"
#include "gpu/cuda/compute/refusal.h"
#include "kernels/spec/resample.h"

namespace ajp::gpu::cuda {

using dev::Elem;
using dev::Plane;
using dev::Tensor;

std::size_t elem_size(Elem elem);
dev::Sample sample_of(kernels::Format format);

// Image YUV dans la mémoire linéaire de l'appareil, à pas de ligne quelconque (NVDEC) : Y, puis
// CbCr entrelacés (NV12, P010, P016) ou Cb puis Cr (formats planaires). planes[2] ne sert pas
// quand la chroma est entrelacée.
struct YuvPlanes {
  kernels::Format format = kernels::Format::Nv12;
  int width = 0;
  int height = 0;
  std::array<Plane, 3> planes{};
};

// Description de couleur déjà résolue (kernels::effective_matrix, effective_range,
// effective_siting).
struct ColorSpec {
  kernels::Matrix matrix = kernels::Matrix::Bt709;
  kernels::Range range = kernels::Range::Limited;
  kernels::Siting siting = kernels::Siting::Left;
};

// Les plans [first, first + count) d'un tenseur, sans copie.
Tensor channels(const Tensor& tensor, int first, int count);

// Dimensions d'une image de la chaîne : paires, dans [2, kMaxDimension].
Result<> check_frame(int width, int height, std::string_view what);

// Un tenseur de width × height × channels : adresse non nulle et alignée sur son élément.
Result<> check_tensor(const Tensor& tensor, int width, int height, int channels, std::string_view what);

// Une image YUV du format et de la taille attendus : chaque plan utile non nul, aligné sur son
// échantillon, de pas multiple de l'échantillon et au moins aussi long qu'une ligne.
Result<> check_yuv(const YuvPlanes& image, kernels::Format format, int width, int height,
                   std::string_view what);

// Le plan de luma seul (détecteurs), de width × height.
Result<> check_luma(const YuvPlanes& image, kernels::Format format, int width, int height,
                    std::string_view what);

}  // namespace ajp::gpu::cuda
