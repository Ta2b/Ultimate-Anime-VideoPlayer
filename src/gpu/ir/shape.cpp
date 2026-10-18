#include "gpu/ir/shape.h"

#include <format>
#include <optional>
#include <utility>
#include <vector>

#include "gpu/ir/graph.h"
#include "gpu/ir/ops.h"

namespace ajp::gpu::ir {
namespace {

// Au-delà, un tenseur d'entiers n'est plus une forme : on ne le propage pas.
constexpr std::size_t kMaxShapeValues = 4096;

std::unexpected<Refusal> shape_invalid(std::string detail) {
  return std::unexpected(Refusal{RefusalCode::ShapeInvalid, std::move(detail)});
}

std::expected<void, Refusal> check_dims(const std::string& name, const Dims& dims) {
  if (dims.size() > kMaxRank) return shape_invalid(std::format("{} : rang {}", name, dims.size()));
  std::uint64_t count = 1;
  for (const std::int64_t dim : dims) {
    if (dim < 0 || dim > kMaxDim) return shape_invalid(std::format("{} : dimension {}", name, dim));
    count *= static_cast<std::uint64_t>(dim);
    if (count > kMaxElements)
      return shape_invalid(std::format("{} : {} trop grand", name, format_dims(dims)));
  }
  return {};
}

// Une forme donnée doit respecter le rang et les dimensions déjà déclarées.
bool compatible(const Tensor& tensor, const Dims& given) {
  if (tensor.known) return tensor.dims == given;
  if (tensor.dims.empty()) return true;  // rang inconnu
  if (tensor.dims.size() != given.size()) return false;
  for (std::size_t k = 0; k < given.size(); ++k) {
    if (tensor.dims[k] >= 0 && tensor.dims[k] != given[k]) return false;
  }
  return true;
}

std::expected<void, Refusal> fix_inputs(Graph& graph, std::span<const InputShape> inputs) {
  for (const InputShape& given : inputs) {
    const auto id = find_tensor(graph, given.name);
    if (!id || !is_graph_input(graph, *id)) {
      return std::unexpected(
          Refusal{RefusalCode::GraphInvalid, std::format("{} n'est pas une entrée", given.name)});
    }
    Tensor& tensor = graph.tensors[*id];
    if (auto ok = check_dims(given.name, given.dims); !ok) return ok;
    if (!compatible(tensor, given.dims)) {
      return shape_invalid(
          std::format("{} : {} contredit {}", given.name, format_dims(given.dims), format_dims(tensor.dims)));
    }
    tensor.dims = given.dims;
    tensor.known = true;
  }
  for (const TensorId id : graph.inputs) {
    const Tensor& tensor = graph.tensors[id];
    if (!tensor.known) {
      return shape_invalid(
          std::format("entrée {} {} : forme à donner", tensor.name, format_dims(tensor.dims)));
    }
    if (auto ok = check_dims(tensor.name, tensor.dims); !ok) return ok;
  }
  return {};
}

}  // namespace

std::expected<void, Refusal> infer_shapes(Graph& graph, std::span<const InputShape> inputs) {
  if (auto ok = fix_inputs(graph, inputs); !ok) return ok;

  // Valeurs des entiers de forme calculées en chemin ; elles ne quittent pas l'inférence.
  std::vector<std::optional<Value>> shape_values(graph.tensors.size());
  for (const Node& node : graph.nodes) {
    std::vector<Operand> operands;
    bool all_known = true;
    for (const TensorId id : node.inputs) {
      if (id == kNoTensor) {
        operands.push_back(Operand{});
        continue;
      }
      const Tensor& tensor = graph.tensors[id];
      const Value* value = tensor.kind == TensorKind::Constant ? &tensor.value
                           : shape_values[id]                  ? &*shape_values[id]
                                                               : nullptr;
      all_known = all_known && value != nullptr;
      operands.push_back(Operand{.type = tensor.type, .dims = &tensor.dims, .value = value});
    }
    auto outs = infer_node(node, operands);
    if (!outs) return std::unexpected(outs.error());

    bool small_ints = true;
    for (std::size_t k = 0; k < node.outputs.size(); ++k) {
      Tensor& out = graph.tensors[node.outputs[k]];
      out.type = (*outs)[k].type;
      out.dims = (*outs)[k].dims;
      out.known = true;
      small_ints = small_ints && out.type == DataType::I64 && element_count(out.dims) <= kMaxShapeValues;
    }
    if (!small_ints) continue;

    // Shape ne lit que la forme de son entrée ; une valeur sans données suffit.
    std::vector<Value> probes;
    std::vector<const Value*> values;
    if (node.op == OpKind::Shape) {
      const Tensor& in = graph.tensors[node.inputs[0]];
      probes.push_back(Value{.type = in.type, .dims = in.dims});
      values.push_back(&probes.back());
    } else if (all_known) {
      for (const Operand& operand : operands) values.push_back(operand.value);
    } else {
      continue;
    }
    auto results = evaluate(node, values);
    if (!results) return std::unexpected(results.error());
    for (std::size_t k = 0; k < node.outputs.size(); ++k)
      shape_values[node.outputs[k]] = std::move((*results)[k]);
  }
  return {};
}

}  // namespace ajp::gpu::ir
