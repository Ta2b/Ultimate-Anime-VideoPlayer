// Numéros de champ d'onnx.proto (ONNX 1.19.1) : ModelProto.graph = 7 ; GraphProto.node = 1, .name = 2,
// .initializer = 5, .input = 11, .output = 12, .sparse_initializer = 15 ; NodeProto.input = 1, .output = 2,
// .name = 3, .op_type = 4, .attribute = 5 ; AttributeProto.name = 1, .i = 3, .s = 4, .t = 5, .g = 6, .graphs
// = 11 ; TensorProto.dims = 1, .data_type = 2, .float_data = 4, .name = 8, .raw_data = 9, .external_data =
// 13, .data_location = 14 ; ValueInfoProto.name = 1, .type = 2 ; TypeProto.tensor_type = 1 ;
// TypeProto.Tensor.elem_type = 1, .shape = 2 ; TensorShapeProto.dim = 1 ; Dimension.dim_value = 1.
#include "prepare/onnx_split.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <format>
#include <set>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "prepare/protobuf_fields.h"

namespace ajp::prepare {
namespace {

using protobuf::Bytes;
using protobuf::Fields;
using protobuf::put_delimited;
using protobuf::put_varint;
using protobuf::RawField;

constexpr std::uint32_t kModelGraph = 7;
constexpr std::uint32_t kGraphNode = 1;
constexpr std::uint32_t kGraphName = 2;
constexpr std::uint32_t kGraphInitializer = 5;
constexpr std::uint32_t kGraphInput = 11;
constexpr std::uint32_t kGraphOutput = 12;
constexpr std::uint32_t kGraphSparse = 15;
constexpr std::uint32_t kNodeInput = 1;
constexpr std::uint32_t kNodeOutput = 2;
constexpr std::uint32_t kNodeName = 3;
constexpr std::uint32_t kNodeOpType = 4;
constexpr std::uint32_t kNodeAttribute = 5;
constexpr std::uint32_t kAttrName = 1;
constexpr std::uint32_t kAttrInteger = 3;
constexpr std::uint32_t kAttrText = 4;
constexpr std::uint32_t kAttrTensor = 5;
constexpr std::uint32_t kAttrGraph = 6;
constexpr std::uint32_t kAttrGraphs = 11;
constexpr std::uint32_t kTensorDims = 1;
constexpr std::uint32_t kTensorType = 2;
constexpr std::uint32_t kTensorFloats = 4;
constexpr std::uint32_t kTensorName = 8;
constexpr std::uint32_t kTensorRaw = 9;
constexpr std::uint32_t kTensorExternal = 13;
constexpr std::uint32_t kTensorLocation = 14;
constexpr std::int32_t kFloat32 = 1;
constexpr std::int32_t kFloat16 = 10;
constexpr std::int32_t kInt64 = 7;
constexpr std::int64_t kElementCap = 1099511627776;  // 2^40 : le produit des dimensions ne déborde pas

std::unexpected<Refusal> malformed(std::string detail) {
  return std::unexpected(Refusal{RefusalCode::ModelUnreadable, std::move(detail)});
}

std::string text_of(Bytes bytes) {
  return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

Bytes as_bytes(std::string_view text) {
  return {reinterpret_cast<const std::byte*>(text.data()), text.size()};
}

double half_to_double(std::uint16_t bits) {
  const std::uint32_t raw = bits;
  const int sign = (raw >> 15U) != 0 ? -1 : 1;
  const auto exponent = static_cast<int>((raw >> 10U) & 0x1FU);
  const auto mantissa = static_cast<int>(raw & 0x3FFU);
  if (exponent == 0) return sign * std::ldexp(static_cast<double>(mantissa), -24);
  if (exponent == 31) return mantissa == 0 ? sign * HUGE_VAL : std::nan("");
  return sign * std::ldexp(1.0 + mantissa / 1024.0, exponent - 15);
}

// Les valeurs d'un TensorProto flottant (32 ou 16 bits) ou entier 64 bits de moins de kMaxConstantElements
// éléments.
struct TensorValues {
  std::string name;
  std::vector<double> values;
  bool external = false;
  bool floating = false;
};

std::expected<TensorValues, Refusal> read_tensor(Bytes message) {
  TensorValues out;
  std::int32_t type = 0;
  std::int64_t elements = 1;
  bool has_dims = false;
  Bytes raw;
  Bytes floats;
  Fields fields(message);
  RawField field;
  while (fields.next(field)) {
    if (field.number == kTensorName && field.delimited) {
      out.name = text_of(field.payload);
    } else if (field.number == kTensorType && !field.delimited) {
      type = static_cast<std::int32_t>(field.scalar);
    } else if (field.number == kTensorRaw && field.delimited) {
      raw = field.payload;
    } else if (field.number == kTensorFloats && field.delimited) {
      floats = field.payload;
    } else if (field.number == kTensorExternal || (field.number == kTensorLocation && field.scalar == 1)) {
      out.external = true;
    } else if (field.number == kTensorDims) {
      has_dims = true;
      if (field.delimited) {  // dims empaquetées
        // Un varint par dimension : relu à la main, Fields ne lit que des champs étiquetés.
        std::size_t pos = 0;
        while (pos < field.payload.size()) {
          std::uint64_t value = 0;
          unsigned shift = 0;
          while (pos < field.payload.size() && shift < 64) {
            const auto byte = std::to_integer<std::uint64_t>(field.payload[pos++]);
            value |= (byte & 0x7fU) << shift;
            shift += 7;
            if ((byte & 0x80U) == 0) break;
          }
          elements *= static_cast<std::int64_t>(value);
          elements = std::min(elements, kElementCap);
        }
      } else {
        elements *= static_cast<std::int64_t>(field.scalar);
        elements = std::min(elements, kElementCap);
      }
    }
  }
  if (!fields.ok()) return malformed("ONNX : tenseur malformé");
  if (!has_dims) elements = 1;  // scalaire
  out.floating = type == kFloat32 || type == kFloat16 || type == kInt64;
  if (!out.floating || elements <= 0 || elements > static_cast<std::int64_t>(kMaxConstantElements))
    return out;
  const auto count = static_cast<std::size_t>(elements);
  if (type == kFloat32 && raw.size() == count * 4) {
    for (std::size_t i = 0; i < count; ++i) {
      std::uint32_t bits = 0;
      std::memcpy(&bits, raw.data() + i * 4, 4);
      out.values.push_back(static_cast<double>(std::bit_cast<float>(bits)));
    }
  } else if (type == kInt64 && raw.size() == count * 8) {
    for (std::size_t i = 0; i < count; ++i) {
      std::int64_t value = 0;
      std::memcpy(&value, raw.data() + i * 8, 8);
      out.values.push_back(static_cast<double>(value));
    }
  } else if (type == kFloat16 && raw.size() == count * 2) {
    for (std::size_t i = 0; i < count; ++i) {
      std::uint16_t bits = 0;
      std::memcpy(&bits, raw.data() + i * 2, 2);
      out.values.push_back(half_to_double(bits));
    }
  } else if (type == kFloat32 && floats.size() == count * 4) {
    for (std::size_t i = 0; i < count; ++i) {
      std::uint32_t bits = 0;
      std::memcpy(&bits, floats.data() + i * 4, 4);
      out.values.push_back(static_cast<double>(std::bit_cast<float>(bits)));
    }
  }
  return out;
}

std::expected<Bytes, Refusal> graph_of(Bytes model) {
  Fields fields(model);
  RawField field;
  std::optional<Bytes> graph;
  while (fields.next(field)) {
    if (field.number == kModelGraph && field.delimited) graph = field.payload;
  }
  if (!fields.ok()) return malformed("ONNX : protobuf du modèle malformé");
  if (!graph) return malformed("ONNX : pas de graphe principal");
  return *graph;
}

// ValueInfoProto { name ; type { tensor_type { elem_type ; shape { dim { dim_value }… } } } }.
std::vector<std::byte> value_info(const SplitTensor& tensor) {
  std::vector<std::byte> inner;
  put_varint(inner, (1U << 3U) | 0U);
  put_varint(inner, static_cast<std::uint64_t>(static_cast<std::uint32_t>(tensor.elem_type)));
  if (!tensor.dims.empty()) {
    std::vector<std::byte> shape;
    for (const std::int64_t dim : tensor.dims) {
      std::vector<std::byte> one;
      put_varint(one, (1U << 3U) | 0U);
      put_varint(one, static_cast<std::uint64_t>(dim));
      put_delimited(shape, 1, one);
    }
    put_delimited(inner, 2, shape);
  }
  std::vector<std::byte> type;
  put_delimited(type, 1, inner);
  std::vector<std::byte> info;
  put_delimited(info, 1, as_bytes(tensor.name));
  put_delimited(info, 2, type);
  return info;
}

}  // namespace

const OnnxAttribute* OnnxFullNode::attribute(std::string_view attribute_name) const {
  const auto found = std::ranges::find(attributes, attribute_name, &OnnxAttribute::name);
  return found == attributes.end() ? nullptr : &*found;
}

std::expected<OnnxGraph, Refusal> read_onnx_graph(std::span<const std::byte> model) {
  auto graph = graph_of(model);
  if (!graph) return std::unexpected(std::move(graph.error()));
  OnnxGraph out;
  Fields fields(*graph);
  RawField field;
  while (fields.next(field)) {
    if (!field.delimited) continue;
    if (field.number == kGraphSparse) {
      out.has_sparse_initializers = true;
    } else if (field.number == kGraphInitializer) {
      auto tensor = read_tensor(field.payload);
      if (!tensor) return std::unexpected(std::move(tensor.error()));
      out.initializers.push_back(tensor->name);
      out.has_external_data = out.has_external_data || tensor->external;
      if (!tensor->values.empty()) out.constants[tensor->name] = std::move(tensor->values);
    } else if (field.number == kGraphNode) {
      OnnxFullNode& node = out.nodes.emplace_back();
      Fields parts(field.payload);
      RawField part;
      while (parts.next(part)) {
        if (!part.delimited) continue;
        if (part.number == kNodeInput) {
          node.inputs.push_back(text_of(part.payload));
        } else if (part.number == kNodeOutput) {
          node.outputs.push_back(text_of(part.payload));
        } else if (part.number == kNodeName) {
          node.name = text_of(part.payload);
        } else if (part.number == kNodeOpType) {
          node.op_type = text_of(part.payload);
        } else if (part.number == kNodeAttribute) {
          OnnxAttribute attribute;
          Fields inner(part.payload);
          RawField item;
          while (inner.next(item)) {
            if (item.number == kAttrName && item.delimited) {
              attribute.name = text_of(item.payload);
            } else if (item.number == kAttrInteger && !item.delimited) {
              attribute.integer = static_cast<std::int64_t>(item.scalar);
            } else if (item.number == kAttrText && item.delimited) {
              attribute.text = text_of(item.payload);
            } else if (item.number == kAttrTensor && item.delimited) {
              auto tensor = read_tensor(item.payload);
              if (!tensor) return std::unexpected(std::move(tensor.error()));
              node.constant = std::move(tensor->values);
              out.has_external_data = out.has_external_data || tensor->external;
            } else if (item.number == kAttrGraph || item.number == kAttrGraphs) {
              out.has_subgraphs = true;
            }
          }
          if (!inner.ok()) return malformed("ONNX : attribut malformé");
          node.attributes.push_back(std::move(attribute));
        }
      }
      if (!parts.ok()) return malformed("ONNX : nœud malformé");
      if (node.op_type == "Constant" && node.outputs.size() == 1 && !node.constant.empty())
        out.constants[node.outputs[0]] = node.constant;
    }
  }
  if (!fields.ok()) return malformed("ONNX : protobuf du graphe malformé");
  return out;
}

std::expected<std::vector<std::byte>, Refusal> extract_subgraph(std::span<const std::byte> model,
                                                                std::span<const SplitTensor> inputs,
                                                                std::span<const SplitTensor> outputs,
                                                                std::string_view graph_name) {
  auto graph = graph_of(model);
  if (!graph) return std::unexpected(std::move(graph.error()));

  struct Node {
    Bytes raw;  // le champ entier, étiquette comprise
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
  };
  struct Initializer {
    Bytes raw;
    std::string name;
  };
  std::vector<Node> nodes;
  std::vector<Initializer> initializers;
  Fields fields(*graph);
  RawField field;
  while (fields.next(field)) {
    if (!field.delimited) continue;
    if (field.number == kGraphSparse)
      return std::unexpected(
          Refusal{RefusalCode::FormatUnsupported, "ONNX : initialiseurs creux non repris"});
    if (field.number == kGraphNode) {
      Node& node = nodes.emplace_back(Node{.raw = field.raw});
      Fields parts(field.payload);
      RawField part;
      while (parts.next(part)) {
        if (part.number == kNodeAttribute && part.delimited) {
          Fields inner(part.payload);
          RawField item;
          while (inner.next(item)) {
            if (item.number == kAttrGraph || item.number == kAttrGraphs)
              return std::unexpected(
                  Refusal{RefusalCode::FormatUnsupported, "ONNX : graphes imbriqués non repris"});
          }
        }
        if (!part.delimited) continue;
        if (part.number == kNodeInput) node.inputs.push_back(text_of(part.payload));
        if (part.number == kNodeOutput) node.outputs.push_back(text_of(part.payload));
      }
      if (!parts.ok()) return malformed("ONNX : nœud malformé");
    } else if (field.number == kGraphInitializer) {
      auto tensor = read_tensor(field.payload);
      if (!tensor) return std::unexpected(std::move(tensor.error()));
      if (tensor->external)
        return std::unexpected(Refusal{RefusalCode::FormatUnsupported, "ONNX : poids externes non repris"});
      initializers.push_back({.raw = field.raw, .name = std::move(tensor->name)});
    }
  }
  if (!fields.ok()) return malformed("ONNX : protobuf du graphe malformé");

  std::unordered_map<std::string, std::size_t> producer;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    for (const std::string& name : nodes[i].outputs) producer.emplace(name, i);
  }
  std::unordered_map<std::string, std::size_t> initializer;
  for (std::size_t i = 0; i < initializers.size(); ++i) initializer.emplace(initializers[i].name, i);
  std::set<std::string, std::less<>> stops;
  for (const SplitTensor& tensor : inputs) stops.insert(tensor.name);

  // Les tenseurs utiles, depuis les sorties en remontant jusqu'aux entrées du morceau.
  std::vector<bool> keep_node(nodes.size(), false);
  std::vector<bool> keep_initializer(initializers.size(), false);
  std::set<std::string, std::less<>> seen;
  std::vector<std::string> pending;
  for (const SplitTensor& tensor : outputs) pending.push_back(tensor.name);
  while (!pending.empty()) {
    std::string name = std::move(pending.back());
    pending.pop_back();
    if (name.empty() || !seen.insert(name).second || stops.contains(name)) continue;
    if (const auto node = producer.find(name); node != producer.end()) {
      if (keep_node[node->second]) continue;
      keep_node[node->second] = true;
      for (const std::string& input : nodes[node->second].inputs) pending.push_back(input);
    } else if (const auto init = initializer.find(name); init != initializer.end()) {
      keep_initializer[init->second] = true;
    } else {
      return std::unexpected(Refusal{
          RefusalCode::FieldMissing,
          std::format("ONNX : « {} » n'est ni une entrée du morceau, ni une constante, ni produit", name)});
    }
  }
  for (const SplitTensor& tensor : outputs) {
    if (stops.contains(tensor.name) || !producer.contains(tensor.name))
      return std::unexpected(Refusal{RefusalCode::FieldMissing,
                                     std::format("ONNX : aucun nœud ne produit « {} »", tensor.name)});
  }

  std::vector<std::byte> new_graph;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (keep_node[i]) new_graph.insert(new_graph.end(), nodes[i].raw.begin(), nodes[i].raw.end());
  }
  put_delimited(new_graph, kGraphName, as_bytes(graph_name));
  for (std::size_t i = 0; i < initializers.size(); ++i) {
    if (keep_initializer[i])
      new_graph.insert(new_graph.end(), initializers[i].raw.begin(), initializers[i].raw.end());
  }
  for (const SplitTensor& tensor : inputs) put_delimited(new_graph, kGraphInput, value_info(tensor));
  for (const SplitTensor& tensor : outputs) put_delimited(new_graph, kGraphOutput, value_info(tensor));

  std::vector<std::byte> out;
  out.reserve(new_graph.size() + 128);
  Fields model_fields(model);
  while (model_fields.next(field)) {
    if (field.number == kModelGraph && field.delimited) {
      put_delimited(out, kModelGraph, new_graph);
    } else {
      out.insert(out.end(), field.raw.begin(), field.raw.end());
    }
  }
  if (!model_fields.ok()) return malformed("ONNX : protobuf du modèle malformé");
  return out;
}

}  // namespace ajp::prepare
