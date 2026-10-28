// Inférence des formes : fixe les dimensions symboliques des entrées, puis
// déduit type et forme de chaque tenseur, nœud après nœud.
#pragma once

#include <expected>
#include <span>
#include <string>

#include "core/types.h"
#include "gpu/ir/ir.h"

namespace ajp::gpu::ir {

struct InputShape {
  std::string name;
  Dims dims;
};

// Les entiers de forme (Shape, Gather, Concat…) sont propagés pendant
// l'inférence, pour connaître les paramètres de Reshape, Slice, Pad ou Resize.
// Une forme déjà connue doit égaler celle donnée ; une entrée dont une
// dimension reste inconnue est refusée (ShapeInvalid).
std::expected<void, Refusal> infer_shapes(Graph& graph, std::span<const InputShape> inputs = {});

}  // namespace ajp::gpu::ir
