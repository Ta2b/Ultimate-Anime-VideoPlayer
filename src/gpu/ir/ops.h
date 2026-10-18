// Table des opérations de l'IR : nom ONNX, nombre d'entrées et de sorties,
// règle de forme et calcul de référence FP32.
//
// La règle de forme et le calcul partagent leur validation : evaluate() appelle
// d'abord la règle, puis calcule sur des sorties déjà dimensionnées.
#pragma once

#include <cstddef>
#include <expected>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "core/types.h"
#include "gpu/ir/ir.h"

namespace ajp::gpu::ir {

// Ce que la règle de forme sait d'une entrée. dims == nullptr : entrée absente.
// value n'est renseignée que si l'entrée est connue avant l'exécution.
struct Operand {
  DataType type = DataType::F32;
  const Dims* dims = nullptr;
  const Value* value = nullptr;
};

struct OutputInfo {
  DataType type = DataType::F32;
  Dims dims;
};

using Inferred = std::expected<std::vector<OutputInfo>, Refusal>;
using Evaluated = std::expected<std::vector<Value>, Refusal>;

using InferFn = Inferred (*)(const Node&, std::span<const Operand>);
// Les entrées sont toutes connues (nullptr : absente) ; outputs vient de la règle de forme.
using EvalFn = Evaluated (*)(const Node&, std::span<const Value* const>, std::span<const OutputInfo>);

struct OpInfo {
  OpKind kind;
  std::string_view name;
  std::size_t min_inputs;
  std::size_t onnx_max_inputs;  // les fusions peuvent en ajouter (Conv)
  std::size_t max_inputs;
  std::size_t min_outputs;
  std::size_t max_outputs;
  InferFn infer;
  EvalFn eval;
};

const OpInfo& op_info(OpKind op);
std::string_view name(OpKind op);
std::optional<OpKind> op_from_onnx(std::string_view op_type);

// Règle de forme du nœud : vérifie l'arité, puis les bornes des formes produites.
Inferred infer_node(const Node& node, std::span<const Operand> inputs);

// Calcul de référence du nœud sur des valeurs connues.
Evaluated evaluate(const Node& node, std::span<const Value* const> inputs);

}  // namespace ajp::gpu::ir
