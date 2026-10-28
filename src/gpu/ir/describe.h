// Descriptions lisibles de l'IR, pour ajp_onnx_info et les journaux.
#pragma once

#include <cstdint>
#include <string>

#include "gpu/ir/ir.h"
#include "gpu/ir/memory_plan.h"
#include "gpu/ir/passes.h"

namespace ajp::gpu::ir {

std::string format_bytes(std::uint64_t bytes);

// « Conv 23, Relu 10, … », du plus fréquent au plus rare.
std::string describe_ops(const Graph& graph);

// Entrées et sorties avec type et forme, une par ligne.
std::string describe_interface(const Graph& graph);

// Une ligne par nœud : opération, paramètres, sorties avec forme et disposition.
std::string describe_nodes(const Graph& graph);

std::string describe_stats(const PassStats& stats);
std::string describe_memory(const MemoryPlan& plan);

}  // namespace ajp::gpu::ir
