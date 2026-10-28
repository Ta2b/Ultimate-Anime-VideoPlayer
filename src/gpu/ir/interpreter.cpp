#include "gpu/ir/interpreter.h"

#include <format>
#include <optional>
#include <utility>

#include "gpu/ir/graph.h"
#include "gpu/ir/ops.h"

namespace ajp::gpu::ir {
namespace {

std::unexpected<Refusal> invalid(std::string detail) {
  return std::unexpected(Refusal{RefusalCode::GraphInvalid, std::move(detail)});
}

std::expected<void, Refusal> check_input(const Tensor& tensor, const Value& value) {
  if (value.type != tensor.type) {
    return invalid(std::format("{} : {} donné pour {}", tensor.name, name(value.type), name(tensor.type)));
  }
  if (!bounded_dims(value.dims))
    return invalid(std::format("{} : forme {} hors des bornes", tensor.name, format_dims(value.dims)));
  // Rang inconnu (aucune dimension déclarée), ou même rang et mêmes dimensions déclarées.
  bool shape_ok = tensor.dims.empty() || tensor.dims.size() == value.dims.size();
  for (std::size_t k = 0; shape_ok && !tensor.dims.empty() && k < value.dims.size(); ++k)
    shape_ok = tensor.dims[k] < 0 || tensor.dims[k] == value.dims[k];
  if ((tensor.known && value.dims != tensor.dims) || !shape_ok) {
    return std::unexpected(
        Refusal{RefusalCode::ShapeInvalid, std::format("{} : {} donné pour {}", tensor.name,
                                                       format_dims(value.dims), format_dims(tensor.dims))});
  }
  if (value_size(value) != element_count(value.dims) ||
      (value.type == DataType::I64 ? !value.f.empty() : !value.i.empty())) {
    return invalid(std::format("{} : données d'une autre taille que la forme", tensor.name));
  }
  return {};
}

}  // namespace

std::expected<std::vector<NamedValue>, Refusal> run(const Graph& graph, std::span<const NamedValue> inputs) {
  // Un graphe construit à la main n'est pas passé par l'import : on ne lui fait pas confiance.
  if (auto ok = validate(graph); !ok) return std::unexpected(ok.error());
  const std::size_t count = graph.tensors.size();
  std::vector<const Value*> values(count, nullptr);
  for (std::size_t i = 0; i < count; ++i) {
    if (graph.tensors[i].kind == TensorKind::Constant) values[i] = &graph.tensors[i].value;
  }
  for (const NamedValue& given : inputs) {
    const auto id = find_tensor(graph, given.name);
    if (!id || !is_graph_input(graph, *id))
      return invalid(std::format("{} n'est pas une entrée du graphe", given.name));
    if (values[*id] != nullptr) return invalid(std::format("{} donné deux fois", given.name));
    if (auto ok = check_input(graph.tensors[*id], given.value); !ok) return std::unexpected(ok.error());
    values[*id] = &given.value;
  }
  for (const TensorId id : graph.inputs) {
    if (values[id] == nullptr) return invalid(std::format("entrée {} absente", graph.tensors[id].name));
  }

  // Dernière lecture de chaque activation ; les sorties du graphe restent jusqu'au bout.
  std::vector<std::size_t> last_use(count, 0);
  for (std::size_t n = 0; n < graph.nodes.size(); ++n) {
    for (const TensorId id : graph.nodes[n].inputs) {
      if (id != kNoTensor) last_use[id] = n;
    }
  }
  for (const TensorId id : graph.outputs) last_use[id] = graph.nodes.size();

  std::vector<std::optional<Value>> computed(count);
  for (std::size_t n = 0; n < graph.nodes.size(); ++n) {
    const Node& node = graph.nodes[n];
    std::vector<const Value*> args;
    for (const TensorId id : node.inputs) {
      if (id != kNoTensor && values[id] == nullptr) {
        return invalid(std::format("{} : {} pas encore calculé", node.name, graph.tensors[id].name));
      }
      args.push_back(id == kNoTensor ? nullptr : values[id]);
    }
    auto results = evaluate(node, args);
    if (!results) return std::unexpected(results.error());
    for (std::size_t k = 0; k < node.outputs.size(); ++k) {
      const TensorId id = node.outputs[k];
      values[id] = &computed[id].emplace(std::move((*results)[k]));
    }
    for (const TensorId id : node.inputs) {
      if (id != kNoTensor && computed[id] && last_use[id] == n) {
        computed[id].reset();
        values[id] = nullptr;
      }
    }
  }

  std::vector<NamedValue> outputs;
  for (const TensorId id : graph.outputs) {
    if (values[id] == nullptr)
      return invalid(std::format("sortie {} jamais calculée", graph.tensors[id].name));
    outputs.push_back({.name = graph.tensors[id].name, .value = *values[id]});
  }
  return outputs;
}

}  // namespace ajp::gpu::ir
