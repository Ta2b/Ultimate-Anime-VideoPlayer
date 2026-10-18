#include "gpu/ir/graph.h"

#include <algorithm>
#include <format>
#include <utility>

namespace ajp::gpu::ir {

std::string_view name(DataType type) {
  switch (type) {
    case DataType::F32:
      return "F32";
    case DataType::F16:
      return "F16";
    case DataType::I64:
      return "I64";
  }
  return "?";
}

std::size_t element_size(DataType type) {
  switch (type) {
    case DataType::F16:
      return 2;
    case DataType::F32:
      return 4;
    case DataType::I64:
      return 8;
  }
  return 0;
}

std::size_t element_count(const Dims& dims) {
  std::size_t count = 1;
  for (const std::int64_t dim : dims) count *= static_cast<std::size_t>(dim);
  return count;
}

std::string format_dims(const Dims& dims) {
  std::string out = "[";
  for (std::size_t i = 0; i < dims.size(); ++i) {
    if (i != 0) out += ',';
    out += dims[i] < 0 ? std::string{"?"} : std::to_string(dims[i]);
  }
  return out + "]";
}

Value make_value(DataType type, Dims dims) {
  Value value{.type = type, .dims = std::move(dims)};
  const std::size_t count = element_count(value.dims);
  if (type == DataType::I64) {
    value.i.assign(count, 0);
  } else {
    value.f.assign(count, 0.0F);
  }
  return value;
}

std::size_t value_size(const Value& value) {
  return value.type == DataType::I64 ? value.i.size() : value.f.size();
}

std::int64_t align_up(std::int64_t value, std::int64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

std::uint64_t storage_bytes(const Tensor& tensor) {
  if (!tensor.known) return 0;
  std::uint64_t count = element_count(tensor.dims);
  if (tensor.layout == Layout::Nhwc16 && tensor.dims.size() == 4 && tensor.dims[1] > 0) {
    const auto channels = static_cast<std::uint64_t>(tensor.dims[1]);
    count = count / channels * static_cast<std::uint64_t>(align_up(tensor.dims[1], 16));
  }
  return count * element_size(tensor.type);
}

TensorId add_tensor(Graph& graph, Tensor tensor) {
  graph.tensors.push_back(std::move(tensor));
  return static_cast<TensorId>(graph.tensors.size() - 1);
}

std::optional<TensorId> find_tensor(const Graph& graph, std::string_view name) {
  for (std::size_t i = 0; i < graph.tensors.size(); ++i) {
    if (graph.tensors[i].name == name) return static_cast<TensorId>(i);
  }
  return std::nullopt;
}

bool is_graph_output(const Graph& graph, TensorId id) {
  return std::ranges::find(graph.outputs, id) != graph.outputs.end();
}

bool is_graph_input(const Graph& graph, TensorId id) {
  return std::ranges::find(graph.inputs, id) != graph.inputs.end();
}

std::vector<std::vector<std::size_t>> consumers(const Graph& graph) {
  std::vector<std::vector<std::size_t>> out(graph.tensors.size());
  for (std::size_t n = 0; n < graph.nodes.size(); ++n) {
    for (const TensorId id : graph.nodes[n].inputs) {
      if (id == kNoTensor) continue;
      auto& list = out[id];
      if (list.empty() || list.back() != n) list.push_back(n);
    }
  }
  return out;
}

std::vector<std::size_t> producers(const Graph& graph) {
  std::vector<std::size_t> out(graph.tensors.size(), kNoNode);
  for (std::size_t n = 0; n < graph.nodes.size(); ++n) {
    for (const TensorId id : graph.nodes[n].outputs) out[id] = n;
  }
  return out;
}

void replace_uses(Graph& graph, TensorId from, TensorId to) {
  for (Node& node : graph.nodes) std::ranges::replace(node.inputs, from, to);
}

void compact(Graph& graph) {
  std::vector<bool> used(graph.tensors.size(), false);
  for (const Node& node : graph.nodes) {
    for (const TensorId id : node.inputs) {
      if (id != kNoTensor) used[id] = true;
    }
    for (const TensorId id : node.outputs) used[id] = true;
  }
  for (const TensorId id : graph.inputs) used[id] = true;
  for (const TensorId id : graph.outputs) used[id] = true;

  std::vector<TensorId> remap(graph.tensors.size(), kNoTensor);
  std::vector<Tensor> kept;
  for (std::size_t i = 0; i < graph.tensors.size(); ++i) {
    if (!used[i]) continue;
    remap[i] = static_cast<TensorId>(kept.size());
    kept.push_back(std::move(graph.tensors[i]));
  }
  auto renumber = [&](std::vector<TensorId>& ids) {
    for (TensorId& id : ids) {
      if (id != kNoTensor) id = remap[id];
    }
  };
  for (Node& node : graph.nodes) {
    renumber(node.inputs);
    renumber(node.outputs);
  }
  renumber(graph.inputs);
  renumber(graph.outputs);
  graph.tensors = std::move(kept);
}

std::expected<void, Refusal> validate(const Graph& graph) {
  auto invalid = [](std::string detail) {
    return std::unexpected(Refusal{RefusalCode::GraphInvalid, std::move(detail)});
  };
  const std::size_t count = graph.tensors.size();
  // Un tenseur est disponible s'il est une entrée, une constante, ou écrit plus haut.
  std::vector<bool> ready(count, false);
  for (std::size_t i = 0; i < count; ++i) ready[i] = graph.tensors[i].kind != TensorKind::Activation;
  for (const TensorId id : graph.inputs) {
    if (id >= count || graph.tensors[id].kind != TensorKind::Input)
      return invalid("entrée de graphe invalide");
  }
  for (const Node& node : graph.nodes) {
    for (const TensorId id : node.inputs) {
      if (id == kNoTensor) continue;
      if (id >= count) return invalid(std::format("{} : entrée hors du graphe", node.name));
      if (!ready[id])
        return invalid(std::format("{} : {} lu avant d'être écrit", node.name, graph.tensors[id].name));
    }
    for (const TensorId id : node.outputs) {
      if (id >= count) return invalid(std::format("{} : sortie hors du graphe", node.name));
      if (ready[id])
        return invalid(std::format("{} : {} écrit deux fois", node.name, graph.tensors[id].name));
      ready[id] = true;
    }
  }
  for (const TensorId id : graph.outputs) {
    if (id >= count || !ready[id]) return invalid("sortie de graphe jamais écrite");
  }
  return {};
}

}  // namespace ajp::gpu::ir
