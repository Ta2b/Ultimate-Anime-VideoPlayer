// Interpréteur CPU FP32 de l'IR : la référence de justesse du moteur Vulkan.
//
// Il exécute les nœuds dans l'ordre du graphe avec les calculs de ops.h, et
// libère chaque activation après sa dernière lecture. Les tenseurs F16 sont
// calculés en FP32 ; seul un Cast vers F16 arrondit.
#pragma once

#include <expected>
#include <span>
#include <string>
#include <vector>

#include "core/types.h"
#include "gpu/ir/ir.h"

namespace ajp::gpu::ir {

struct NamedValue {
  std::string name;
  Value value;
};

// Chaque entrée du graphe est donnée une fois, avec son type et sa forme.
// Les sorties reviennent dans l'ordre de graph.outputs.
std::expected<std::vector<NamedValue>, Refusal> run(const Graph& graph, std::span<const NamedValue> inputs);

}  // namespace ajp::gpu::ir
