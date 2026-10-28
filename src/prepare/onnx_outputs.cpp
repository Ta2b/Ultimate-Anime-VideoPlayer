// Numéros de champ d'onnx.proto (ONNX 1.19.1) : ModelProto.graph = 7 ; GraphProto.node = 1,
// GraphProto.output = 12 ; NodeProto.input = 1, .output = 2, .op_type = 4 ; ValueInfoProto.name = 1, .type =
// 2 ; TypeProto.tensor_type = 1 ; TypeProto.Tensor.elem_type = 1.
#include "prepare/onnx_outputs.h"

#include <algorithm>
#include <format>
#include <optional>
#include <string_view>
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
constexpr std::uint32_t kGraphOutput = 12;
constexpr std::uint32_t kNodeInput = 1;
constexpr std::uint32_t kNodeOutput = 2;
constexpr std::uint32_t kNodeOpType = 4;

std::unexpected<Refusal> malformed(std::string detail) {
  return std::unexpected(Refusal{RefusalCode::ModelUnreadable, std::move(detail)});
}

Bytes as_bytes(std::string_view text) {
  return {reinterpret_cast<const std::byte*>(text.data()), text.size()};
}

// ValueInfoProto { name ; type { tensor_type { elem_type } } } : la forme est laissée à TensorRT.
std::vector<std::byte> value_info(const OnnxOutput& output) {
  std::vector<std::byte> tensor;
  put_varint(tensor, (1U << 3U) | 0U);
  put_varint(tensor, static_cast<std::uint64_t>(static_cast<std::uint32_t>(output.elem_type)));
  std::vector<std::byte> type;
  put_delimited(type, 1, tensor);
  std::vector<std::byte> info;
  put_delimited(info, 1, as_bytes(output.name));
  put_delimited(info, 2, type);
  return info;
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

}  // namespace

std::expected<std::vector<std::string>, Refusal> onnx_node_outputs(std::span<const std::byte> model) {
  auto graph = graph_of(model);
  if (!graph) return std::unexpected(std::move(graph.error()));
  std::vector<std::string> names;
  Fields fields(*graph);
  RawField field;
  while (fields.next(field)) {
    if (field.number != kGraphNode || !field.delimited) continue;
    Fields node(field.payload);
    RawField part;
    while (node.next(part)) {
      if (part.number == kNodeOutput && part.delimited)
        names.emplace_back(reinterpret_cast<const char*>(part.payload.data()), part.payload.size());
    }
    if (!node.ok()) return malformed("ONNX : nœud malformé");
  }
  if (!fields.ok()) return malformed("ONNX : protobuf du graphe malformé");
  return names;
}

std::expected<std::vector<OnnxNode>, Refusal> onnx_nodes(std::span<const std::byte> model) {
  auto graph = graph_of(model);
  if (!graph) return std::unexpected(std::move(graph.error()));
  std::vector<OnnxNode> nodes;
  Fields fields(*graph);
  RawField field;
  while (fields.next(field)) {
    if (field.number != kGraphNode || !field.delimited) continue;
    OnnxNode& node = nodes.emplace_back();
    Fields parts(field.payload);
    RawField part;
    while (parts.next(part)) {
      if (!part.delimited) continue;
      std::string text(reinterpret_cast<const char*>(part.payload.data()), part.payload.size());
      if (part.number == kNodeInput)
        node.inputs.push_back(std::move(text));
      else if (part.number == kNodeOutput)
        node.outputs.push_back(std::move(text));
      else if (part.number == kNodeOpType)
        node.op_type = std::move(text);
    }
    if (!parts.ok()) return malformed("ONNX : nœud malformé");
  }
  if (!fields.ok()) return malformed("ONNX : protobuf du graphe malformé");
  return nodes;
}

std::expected<std::vector<std::byte>, Refusal> with_outputs(std::span<const std::byte> model,
                                                            std::span<const OnnxOutput> outputs) {
  auto produced = onnx_node_outputs(model);
  if (!produced) return std::unexpected(std::move(produced.error()));
  for (const OnnxOutput& output : outputs) {
    if (std::ranges::find(*produced, output.name) == produced->end())
      return std::unexpected(Refusal{RefusalCode::FieldMissing,
                                     std::format("ONNX : aucun nœud ne produit « {} »", output.name)});
  }
  auto graph = graph_of(model);
  if (!graph) return std::unexpected(std::move(graph.error()));

  // Le graphe sans ses sorties, puis les nouvelles, à la fin (l'ordre des champs est libre).
  std::vector<std::byte> new_graph;
  new_graph.reserve(graph->size() + 256);
  Fields graph_fields(*graph);
  RawField field;
  while (graph_fields.next(field)) {
    if (field.number != kGraphOutput) new_graph.insert(new_graph.end(), field.raw.begin(), field.raw.end());
  }
  for (const OnnxOutput& output : outputs) put_delimited(new_graph, kGraphOutput, value_info(output));

  // Le modèle avec ce graphe à la place de l'ancien, le reste tel quel et dans le même ordre.
  std::vector<std::byte> out;
  out.reserve(model.size() + 256);
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
