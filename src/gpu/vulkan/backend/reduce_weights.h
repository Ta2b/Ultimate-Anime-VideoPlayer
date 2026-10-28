// Tables de poids de la réduction fusionnée à l'ingestion (gpu/kernel_ops.h, IngestSpec.reduce),
// pour les passes séparables de vk_compute : le même filtre que le backend TensorRT
// (slice_kernels.cu, filtered_at et area_at), calculé une fois sur l'hôte.
#pragma once

#include <expected>

#include "core/types.h"
#include "gpu/kernel_ops.h"
#include "gpu/vulkan/compute/weights.h"

namespace ajp::gpu::vk {

// dst sorties sur src entrées, dst ≤ src. Area : moyenne exacte de l'empreinte (compute_weights).
// Triangle, Cubic (B, C) et Spline36 : noyau étiré à l'échelle s = src / dst autour du centre
// (x + 0,5) s − 0,5, rayon 1, 2 ou 3 × s, poids normalisés, bords répétés. Les indices répétés
// sont repliés dans une fenêtre contiguë de la source : le miroir des passes n'agit jamais.
// Refus : ShapeInvalid (dst hors de [1, src]), FieldOutOfRange (B ou C non fini).
std::expected<WeightTable, Refusal> reduction_weights(int src, int dst, const Reduction& reduction);

}  // namespace ajp::gpu::vk
