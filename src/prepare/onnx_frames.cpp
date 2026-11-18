// Numéros de champ d'onnx.proto (ONNX 1.19.1) : ModelProto.graph = 7 ; GraphProto.node = 1, .input = 11 ;
// NodeProto.input = 1, .output = 2, .name = 3, .op_type = 4, .attribute = 5 ; AttributeProto.name = 1, .i =
// 3, .type = 20 (INT = 2) ; ValueInfoProto.name = 1, .type = 2 ; TypeProto.tensor_type = 1 ;
// TypeProto.Tensor.elem_type = 1, .shape = 2 ; TensorShapeProto.dim = 1 ; Dimension.dim_value = 1.
#include "prepare/onnx_frames.h"

#include <algorithm>
#include <format>
#include <utility>

#include "prepare/onnx_scan.h"
#include "prepare/onnx_split.h"
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
constexpr std::uint32_t kGraphInput = 11;
constexpr std::uint32_t kValueInfoName = 1;
constexpr std::uint64_t kAttributeInt = 2;

Bytes as_bytes(std::string_view text) {
  return {reinterpret_cast<const std::byte*>(text.data()), text.size()};
}

void put_scalar(std::vector<std::byte>& out, std::uint32_t number, std::uint64_t value) {
  put_varint(out, (std::uint64_t{number} << 3U) | 0U);
  put_varint(out, value);
}

// ValueInfoProto { name ; type { tensor_type { elem_type ; shape { dim { dim_value }… } } } }.
std::vector<std::byte> value_info(std::string_view name, std::int32_t elem_type,
                                  std::span<const std::int64_t> dims) {
  std::vector<std::byte> shape;
  for (const std::int64_t d : dims) {
    std::vector<std::byte> dim;
    put_scalar(dim, 1, static_cast<std::uint64_t>(d));
    put_delimited(shape, 1, dim);
  }
  std::vector<std::byte> tensor;
  put_scalar(tensor, 1, static_cast<std::uint64_t>(static_cast<std::uint32_t>(elem_type)));
  put_delimited(tensor, 2, shape);
  std::vector<std::byte> type;
  put_delimited(type, 1, tensor);
  std::vector<std::byte> info;
  put_delimited(info, kValueInfoName, as_bytes(name));
  put_delimited(info, 2, type);
  return info;
}

// NodeProto Concat(axis = 1) des entrées coupées vers input.
std::vector<std::byte> concat_node(std::string_view input, int frames) {
  std::vector<std::byte> node;
  for (int k = 0; k < frames; ++k) put_delimited(node, 1, as_bytes(frame_input_name(input, k)));
  put_delimited(node, 2, as_bytes(input));
  put_delimited(node, 3, as_bytes(std::format("ajp_frames_{}", input)));
  put_delimited(node, 4, as_bytes("Concat"));
  std::vector<std::byte> axis;
  put_delimited(axis, 1, as_bytes("axis"));
  put_scalar(axis, 3, 1);
  put_scalar(axis, 20, kAttributeInt);
  put_delimited(node, 5, axis);
  return node;
}

// Le nom d'un ValueInfoProto, vide s'il n'en a pas.
std::string_view info_name(Bytes info) {
  Fields fields(info);
  RawField part;
  while (fields.next(part)) {
    if (part.number == kValueInfoName && part.delimited)
      return {reinterpret_cast<const char*>(part.payload.data()), part.payload.size()};
  }
  return {};
}

}  // namespace

std::string frame_input_name(std::string_view input, int k) {
  return std::format("{}_{}", input, k);
}

std::expected<std::vector<std::byte>, Refusal> with_frame_inputs(std::span<const std::byte> model,
                                                                 std::string_view input, int frames) {
  if (frames < 2)
    return refuse(RefusalCode::FieldOutOfRange, std::format("ONNX : {} image(s) à couper", frames));
  auto summary = scan_onnx(model);
  if (!summary) return std::unexpected(std::move(summary.error()));
  const auto found = std::ranges::find(summary->inputs, input, &OnnxInput::name);
  if (found == summary->inputs.end())
    return refuse(RefusalCode::FieldMissing, std::format("ONNX : pas d'entrée « {} »", input));
  if (!found->is_static() || found->dims.size() != 4 || found->dims[1] % frames != 0) {
    return refuse(
        RefusalCode::FormatUnsupported,
        std::format("ONNX : l'entrée « {} » n'est pas une pile statique N × {}·C × H × W", input, frames));
  }
  std::vector<std::int64_t> dims = found->dims;
  dims[1] /= frames;
  for (int k = 0; k < frames; ++k) {
    if (std::ranges::contains(summary->inputs, frame_input_name(input, k), &OnnxInput::name))
      return refuse(RefusalCode::FormatUnsupported,
                    std::format("ONNX : une entrée s'appelle déjà « {} »", frame_input_name(input, k)));
  }

  auto graph = onnx_graph_field(model);
  if (!graph) return std::unexpected(std::move(graph.error()));
  // Le Concat d'abord : les nœuds restent dans l'ordre d'un tri topologique.
  std::vector<std::byte> new_graph;
  new_graph.reserve(graph->size() + 512);
  put_delimited(new_graph, kGraphNode, concat_node(input, frames));
  Fields graph_fields(*graph);
  RawField field;
  while (graph_fields.next(field)) {
    if (field.number == kGraphInput && field.delimited && info_name(field.payload) == input) continue;
    new_graph.insert(new_graph.end(), field.raw.begin(), field.raw.end());
  }
  if (!graph_fields.ok()) return refuse(RefusalCode::ModelUnreadable, "ONNX : protobuf du graphe malformé");
  for (int k = 0; k < frames; ++k)
    put_delimited(new_graph, kGraphInput, value_info(frame_input_name(input, k), found->elem_type, dims));

  std::vector<std::byte> out;
  out.reserve(model.size() + 512);
  Fields model_fields(model);
  while (model_fields.next(field)) {
    if (field.number == kModelGraph && field.delimited) {
      put_delimited(out, kModelGraph, new_graph);
    } else {
      out.insert(out.end(), field.raw.begin(), field.raw.end());
    }
  }
  if (!model_fields.ok()) return refuse(RefusalCode::ModelUnreadable, "ONNX : protobuf du modèle malformé");
  return out;
}

}  // namespace ajp::prepare
