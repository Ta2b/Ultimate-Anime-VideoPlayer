#include "gpu/ir/import.h"

#include <algorithm>
#include <format>
#include <fstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "gpu/ir/graph.h"
#include "gpu/ir/import_attrs.h"
#include "gpu/ir/onnx_fp8.h"
#include "gpu/ir/ops.h"

namespace ajp::gpu::ir {
namespace {

std::unexpected<Refusal> invalid(std::string detail) {
  return std::unexpected(Refusal{RefusalCode::GraphInvalid, std::move(detail)});
}

std::unexpected<Refusal> unsupported(std::string detail) {
  return std::unexpected(Refusal{RefusalCode::OnnxUnsupported, std::move(detail)});
}

bool default_domain(std::string_view domain) {
  return domain.empty() || domain == "ai.onnx";
}

std::expected<std::int64_t, Refusal> default_opset(const onnx::Model& model) {
  std::optional<std::int64_t> version;
  for (const onnx::OpsetId& opset : model.opsets) {
    if (default_domain(opset.domain)) version = opset.version;
  }
  if (!version) return unsupported("opset du domaine ONNX par défaut absent");
  if (*version < kMinOpset || *version > kMaxOpset) {
    return unsupported(
        std::format("opset {} : seuls {} à {} sont pris en charge", *version, kMinOpset, kMaxOpset));
  }
  return *version;
}

std::string node_label(const onnx::Node& node, std::size_t index) {
  return node.name.empty() ? std::format("{}#{}", node.op_type, index) : node.name;
}

class Builder {
 public:
  std::expected<TensorId, Refusal> declare(Tensor tensor) {
    if (tensor.name.empty()) return invalid("tenseur sans nom");
    if (ids_.contains(tensor.name)) return invalid(std::format("{} déclaré deux fois", tensor.name));
    std::string key = tensor.name;
    const TensorId id = add_tensor(graph, std::move(tensor));
    ids_.emplace(std::move(key), id);
    return id;
  }

  std::expected<TensorId, Refusal> lookup(const std::string& name) const {
    const auto it = ids_.find(name);
    if (it == ids_.end()) return invalid(std::format("tenseur {} jamais défini", name));
    return it->second;
  }

  bool contains(const std::string& name) const { return ids_.contains(name); }

  Graph graph;

 private:
  std::unordered_map<std::string, TensorId> ids_;
};

std::expected<TensorId, Refusal> declare_constant(Builder& b, std::string name,
                                                  std::expected<Value, Refusal> value) {
  if (!value) return std::unexpected(value.error());
  Tensor tensor{.name = std::move(name),
                .kind = TensorKind::Constant,
                .type = value->type,
                .dims = value->dims,
                .known = true,
                .value = std::move(*value)};
  return b.declare(std::move(tensor));
}

std::expected<void, Refusal> declare_input(Builder& b, const onnx::ValueInfo& info) {
  auto type = detail::data_type_of(info.elem_type, info.name);
  if (!type) return std::unexpected(type.error());
  if (info.dims.size() > kMaxRank) return invalid(std::format("{} : rang {}", info.name, info.dims.size()));
  const bool known = info.has_shape && std::ranges::none_of(info.dims, [](std::int64_t d) { return d < 0; });
  Tensor tensor{
      .name = info.name, .kind = TensorKind::Input, .type = *type, .dims = info.dims, .known = known};
  auto id = b.declare(std::move(tensor));
  if (!id) return std::unexpected(id.error());
  b.graph.inputs.push_back(*id);
  return {};
}

std::expected<Node, Refusal> import_node(Builder& b, const onnx::Node& source, std::size_t index,
                                         std::int64_t opset) {
  const std::string label = node_label(source, index);
  if (!default_domain(source.domain)) {
    return std::unexpected(
        Refusal{RefusalCode::OperatorUnknown, std::format("{}:{}", source.domain, source.op_type)});
  }
  const auto op = op_from_onnx(source.op_type);
  if (!op) return std::unexpected(Refusal{RefusalCode::OperatorUnknown, source.op_type});
  if (source.inputs.size() > op_info(*op).onnx_max_inputs) {
    return invalid(std::format("{} : {} entrées pour {}", label, source.inputs.size(), source.op_type));
  }

  auto attrs = detail::import_attrs(*op, source, opset);
  if (!attrs) return std::unexpected(attrs.error());
  Node node{.name = label, .op = *op, .attrs = std::move(attrs->attrs)};
  for (const std::string& name : source.inputs) {
    if (name.empty()) {
      node.inputs.push_back(kNoTensor);
      continue;
    }
    auto id = b.lookup(name);
    if (!id) return std::unexpected(id.error());
    node.inputs.push_back(*id);
  }
  if (attrs->input_from_attr) {
    if (node.inputs.size() > 1) return invalid(std::format("{} : attribut et entrée à la fois", label));
    if (node.inputs.empty()) return invalid(std::format("{} : entrée de données absente", label));
    std::string name = std::format("{}/attribut", label);
    while (b.contains(name)) name += '_';
    auto id = declare_constant(b, name, std::move(*attrs->input_from_attr));
    if (!id) return std::unexpected(id.error());
    node.inputs.push_back(*id);
  }
  while (!node.inputs.empty() && node.inputs.back() == kNoTensor) node.inputs.pop_back();
  for (const std::string& name : source.outputs) {
    auto id = b.lookup(name);
    if (!id) return std::unexpected(id.error());
    node.outputs.push_back(*id);
  }
  return node;
}

// Tri topologique stable : un nœud vient dès que toutes ses entrées existent.
std::expected<std::vector<Node>, Refusal> sort_nodes(const Graph& graph, std::vector<Node> nodes) {
  std::vector<bool> ready(graph.tensors.size(), false);
  for (std::size_t i = 0; i < graph.tensors.size(); ++i)
    ready[i] = graph.tensors[i].kind != TensorKind::Activation;
  std::vector<Node> sorted;
  std::vector<bool> placed(nodes.size(), false);
  while (sorted.size() < nodes.size()) {
    const std::size_t before = sorted.size();
    for (std::size_t n = 0; n < nodes.size(); ++n) {
      if (placed[n]) continue;
      const bool inputs_ready =
          std::ranges::all_of(nodes[n].inputs, [&](TensorId id) { return id == kNoTensor || ready[id]; });
      if (!inputs_ready) continue;
      for (const TensorId id : nodes[n].outputs) ready[id] = true;
      placed[n] = true;
      sorted.push_back(std::move(nodes[n]));
    }
    if (sorted.size() == before) return invalid("le graphe contient un cycle");
  }
  return sorted;
}

}  // namespace

std::expected<Graph, Refusal> import_model(const onnx::Model& model) {
  auto opset = default_opset(model);
  if (!opset) return std::unexpected(opset.error());
  // Les paires Q/DQ FP8 deviennent des FakeQuantE4M3 sur une copie du graphe, jamais sur l'original.
  onnx::Graph fused;
  const bool quantized = std::ranges::any_of(
      model.graph.nodes, [](const onnx::Node& n) { return n.op_type == "QuantizeLinear"; });
  if (quantized) {
    fused = model.graph;
    onnx::fuse_fp8_quant_pairs(fused);
  }
  const onnx::Graph& source = quantized ? fused : model.graph;
  Builder b;
  b.graph.opset = *opset;

  for (const onnx::Tensor& tensor : source.initializers) {
    // Un zéro FP8 resté après la fusion appartient à un Q/DQ non fusionné : le nœud est refusé
    // sous son nom d'opérateur, plus parlant que le type du zéro.
    if (tensor.data_type == onnx::kFloat8E4M3Fn) continue;
    if (auto ok = declare_constant(b, tensor.name, detail::tensor_value(tensor)); !ok)
      return std::unexpected(ok.error());
  }
  for (const onnx::ValueInfo& info : source.inputs) {
    if (b.contains(info.name)) continue;  // IR < 4 : les poids figurent aussi parmi les entrées
    if (auto ok = declare_input(b, info); !ok) return std::unexpected(ok.error());
  }
  // Les sorties des nœuds d'abord, pour accepter un fichier dont l'ordre n'est pas topologique.
  for (std::size_t n = 0; n < source.nodes.size(); ++n) {
    const onnx::Node& node = source.nodes[n];
    if (node.op_type == "Constant" && default_domain(node.domain)) {
      if (node.outputs.size() != 1) return invalid(std::format("{} : une sortie", node_label(node, n)));
      if (auto ok = declare_constant(b, node.outputs[0], detail::constant_value(node)); !ok) {
        return std::unexpected(ok.error());
      }
      continue;
    }
    for (const std::string& name : node.outputs) {
      auto id = b.declare(Tensor{.name = name});
      if (!id) return std::unexpected(id.error());
    }
  }

  std::vector<Node> nodes;
  for (std::size_t n = 0; n < source.nodes.size(); ++n) {
    const onnx::Node& node = source.nodes[n];
    if (node.op_type == "Constant" && default_domain(node.domain)) continue;
    auto imported = import_node(b, node, n, *opset);
    if (!imported) return std::unexpected(imported.error());
    nodes.push_back(std::move(*imported));
  }
  auto sorted = sort_nodes(b.graph, std::move(nodes));
  if (!sorted) return std::unexpected(sorted.error());
  b.graph.nodes = std::move(*sorted);

  for (const onnx::ValueInfo& info : source.outputs) {
    auto id = b.lookup(info.name);
    if (!id) return std::unexpected(id.error());
    b.graph.outputs.push_back(*id);
  }
  if (b.graph.outputs.empty()) return invalid("graphe sans sortie");
  if (auto ok = validate(b.graph); !ok) return std::unexpected(ok.error());
  return std::move(b.graph);
}

std::expected<Graph, Refusal> load_onnx(std::span<const std::byte> bytes) {
  auto model = onnx::read_onnx(bytes);
  if (!model) return std::unexpected(model.error());
  return import_model(*model);
}

std::expected<Graph, Refusal> load_onnx_file(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file)
    return std::unexpected(Refusal{RefusalCode::OnnxMalformed, std::format("{} illisible", path.string())});
  const std::streamoff size = file.tellg();
  // Un message protobuf ne dépasse pas 2 Gio.
  if (size < 0 || size > std::streamoff{0x7FFFFFFF}) {
    return unsupported(std::format("{} : taille hors de [0, 2 Gio]", path.string()));
  }
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(bytes.data()), size);
  if (!file)
    return std::unexpected(
        Refusal{RefusalCode::OnnxMalformed, std::format("{} : lecture tronquée", path.string())});
  return load_onnx(bytes);
}

}  // namespace ajp::gpu::ir
