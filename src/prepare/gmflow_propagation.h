#pragma once

#include <cstddef>
#include <expected>
#include <span>

#include "core/types.h"
#include "prepare/gmflow_propagation_plan.h"

namespace kirari::prepare {

// Lit les constantes du MatMul + Add dans la paire ONNX et vérifie leur partage entre échelles.
// Au huitième k se projette à partir de q, au quart depuis les caractéristiques, comme q.
std::expected<GmflowPropagation, Refusal> gmflow_propagation(std::span<const std::byte> pair);

}  // namespace kirari::prepare
