// Entrée et sortie de RIFE (kernels/spec/rife.h) : les plans constants et l'instant t du tenseur
// d'entrée à 11 plans, et le recadrage de la sortie remplie. L'image elle-même entre par
// IngestPlan (rife_ingest_spec) et sort en YUV par OutputPlan (rife_output_spec), dans yuv.h.
#pragma once

#include <cuda_runtime_api.h>

#include "gpu/cuda/compute/image.h"
#include "kernels/spec/rife.h"

namespace ajp::gpu::cuda {

// Plans 7 à 10 de l'entrée (grilles et pas), une fois par taille : ils ne changent pas d'une
// paire à l'autre. input : 11 plans, de dimensions multiples de 64.
Result<> rife_constants(const Tensor& input, cudaStream_t stream);

// Plan 6 de l'entrée : l'instant t, dans [0, 1].
Result<> rife_time(const Tensor& input, float t, cudaStream_t stream);

// La fenêtre de kernels::rife_geometry(out.width, out.height) des trois plans de padded
// (kernels::ref::rife_crop sur RGB).
Result<> rife_crop(const Tensor& padded, const Tensor& out, cudaStream_t stream);

}  // namespace ajp::gpu::cuda
