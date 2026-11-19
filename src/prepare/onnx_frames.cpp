// Numéros de champ d'onnx.proto (ONNX 1.19.1) : ModelProto.graph = 7 ; GraphProto.node = 1, .initializer = 5,
// .input = 11, .output = 12, .value_info = 13 ; NodeProto.input = 1, .output = 2, .name = 3, .op_type = 4,
// .attribute = 5 ; AttributeProto.name = 1, .i = 3, .s = 4, .ints = 8, .type = 20 (INT = 2, STRING = 3,
// INTS = 7) ; TensorProto.dims = 1, .data_type = 2 (FLOAT = 1), .name = 8, .raw_data = 9 ;
// ValueInfoProto.name = 1, .type = 2 ; TypeProto.tensor_type = 1 ; TypeProto.Tensor.elem_type = 1, .shape = 2
// ; TensorShapeProto.dim = 1 ; Dimension.dim_value = 1, .dim_param = 2.
#include "prepare/onnx_frames.h"

#include <algorithm>
#include <array>
#include <bit>
#include <format>
#include <utility>

#include "prepare/onnx_build.h"
#include "prepare/onnx_scan.h"
#include "prepare/onnx_split.h"
#include "prepare/protobuf_fields.h"

namespace kirari::prepare {
namespace {

using onnx_build::as_text;
using onnx_build::int_attribute;
using onnx_build::kNodeAttribute;
using onnx_build::kNodeInput;
using onnx_build::kNodeName;
using onnx_build::kNodeOpType;
using onnx_build::kNodeOutput;
using onnx_build::kValueInfoName;
using onnx_build::Message;
using onnx_build::node;
using onnx_build::pool_node;
using onnx_build::text_attribute;
using onnx_build::value_info;
using protobuf::Bytes;
using protobuf::Fields;
using protobuf::RawField;

constexpr std::uint32_t kModelGraph = 7;
constexpr std::uint32_t kGraphNode = 1;
constexpr std::uint32_t kGraphInitializer = 5;
constexpr std::uint32_t kGraphInput = 11;
constexpr std::uint32_t kGraphOutput = 12;
constexpr std::uint32_t kGraphValueInfo = 13;

// Les noms que la demi-taille ajoute au graphe.
constexpr std::string_view kFull = "kirari_frames_full";
constexpr std::string_view kNetOut = "kirari_frames_net_out";
constexpr std::string_view kCenterHalf = "kirari_frames_center_half";
constexpr std::string_view kDeltaHalf = "kirari_frames_delta_half";
constexpr std::string_view kDelta = "kirari_frames_delta";
constexpr std::string_view kScales = "kirari_frames_scales";

// Les échelles du Resize, [1, 1, 2, 2] en float32 petit-boutiste.
Message scales_initializer() {
  constexpr std::array<float, 4> kScaleValues = {1.0F, 1.0F, 2.0F, 2.0F};
  std::array<std::byte, sizeof(kScaleValues)> raw{};
  for (std::size_t i = 0; i < kScaleValues.size(); ++i) {
    const auto bits = std::bit_cast<std::uint32_t>(kScaleValues[i]);
    for (std::size_t b = 0; b < 4; ++b) raw[i * 4 + b] = static_cast<std::byte>((bits >> (8U * b)) & 0xFFU);
  }
  Message t;
  t.varint(1, 4).varint(2, 1).text(8, kScales).bytes(9, raw);
  return t;
}

// Les nœuds ajoutés en tête : le Concat des images vers input ; en demi-taille, vers kFull, moyenné vers
// input, et l'image du milieu moyennée.
void head_nodes(Message& graph, const FrameSplit& split) {
  std::vector<std::string> names;
  names.reserve(static_cast<std::size_t>(split.frames));
  for (int k = 0; k < split.frames; ++k) names.push_back(frame_input_name(split.input, k));
  Message concat;
  for (const std::string& name : names) concat.text(kNodeInput, name);
  concat.text(kNodeOutput, split.half ? kFull : split.input)
      .text(kNodeName, std::format("kirari_frames_{}", split.input))
      .text(kNodeOpType, "Concat")
      .message(kNodeAttribute, int_attribute("axis", 1));
  graph.message(kGraphNode, concat);
  if (!split.half) return;
  graph.message(kGraphNode, pool_node("kirari_frames_pool", kFull, split.input));
  graph.message(kGraphNode, pool_node("kirari_frames_center_pool",
                                      names[static_cast<std::size_t>(split.center)], kCenterHalf));
}

// Les nœuds ajoutés en queue (demi-taille) : output = milieu + agrandi(réseau − milieu moyenné).
void tail_nodes(Message& graph, const FrameSplit& split) {
  const std::string center = frame_input_name(split.input, split.center);
  graph.message(kGraphNode,
                node("Sub", "kirari_frames_delta_half", std::array{kNetOut, kCenterHalf}, kDeltaHalf));
  Message resize =
      node("Resize", "kirari_frames_resize", std::array{kDeltaHalf, std::string_view(), kScales}, kDelta);
  resize.message(kNodeAttribute, text_attribute("mode", "linear"));
  graph.message(kGraphNode, resize);
  graph.message(kGraphNode,
                node("Add", "kirari_frames_add", std::array{std::string_view(center), kDelta}, split.output));
  graph.message(kGraphInitializer, scales_initializer());
}

// Le nom d'un ValueInfoProto, vide s'il n'en a pas.
std::string_view info_name(Bytes info) {
  Fields fields(info);
  RawField part;
  while (fields.next(part)) {
    if (part.number == kValueInfoName && part.delimited) return as_text(part.payload);
  }
  return {};
}

// Le nœud, ses entrées et sorties nommées output renommées kNetOut.
std::expected<Message, Refusal> renamed_output(Bytes node_bytes, std::string_view output, bool& renamed) {
  Message out;
  Fields node_fields(node_bytes);
  RawField part;
  while (node_fields.next(part)) {
    const bool io = (part.number == kNodeInput || part.number == kNodeOutput) && part.delimited;
    if (io && as_text(part.payload) == output) {
      out.text(part.number, kNetOut);
      renamed = true;
    } else {
      out.raw(part.raw);
    }
  }
  if (!node_fields.ok()) return refuse(RefusalCode::ModelUnreadable, "ONNX : nœud malformé");
  return out;
}

// Le graphe recopié derrière les nœuds de tête : sans l'entrée coupée ; en demi-taille, sans ses value_info
// et la sortie du réseau renommée kNetOut (output_found : la sortie du graphe est là ; renamed : un nœud la
// produisait).
std::expected<void, Refusal> copy_graph(Bytes graph, const FrameSplit& split, Message& new_graph,
                                        bool& output_found, bool& renamed) {
  Fields graph_fields(graph);
  RawField field;
  while (graph_fields.next(field)) {
    if (field.number == kGraphInput && field.delimited && info_name(field.payload) == split.input) continue;
    if (split.half && field.number == kGraphValueInfo) continue;  // formes d'avant la demi-taille
    if (split.half && field.number == kGraphOutput && field.delimited &&
        info_name(field.payload) == split.output)
      output_found = true;
    if (split.half && field.number == kGraphNode && field.delimited) {
      auto node_bytes = renamed_output(field.payload, split.output, renamed);
      if (!node_bytes) return std::unexpected(std::move(node_bytes.error()));
      new_graph.message(kGraphNode, *node_bytes);
      continue;
    }
    new_graph.raw(field.raw);
  }
  if (!graph_fields.ok()) return refuse(RefusalCode::ModelUnreadable, "ONNX : protobuf du graphe malformé");
  return {};
}

std::expected<void, Refusal> check_split(const OnnxSummary& summary, const FrameSplit& split,
                                         const OnnxInput*& found) {
  if (split.frames < 2)
    return refuse(RefusalCode::FieldOutOfRange, std::format("ONNX : {} image(s) à couper", split.frames));
  if (split.half && (split.center < 0 || split.center >= split.frames))
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("ONNX : image du milieu {} hors des {} images", split.center, split.frames));
  const auto it = std::ranges::find(summary.inputs, split.input, &OnnxInput::name);
  if (it == summary.inputs.end())
    return refuse(RefusalCode::FieldMissing, std::format("ONNX : pas d'entrée « {} »", split.input));
  found = &*it;
  if (found->dims.size() != 4 || found->dims[1] <= 0 || found->dims[1] % split.frames != 0) {
    return refuse(
        RefusalCode::FormatUnsupported,
        std::format("ONNX : l'entrée « {} » n'est pas une pile N × {}·C × H × W", split.input, split.frames));
  }
  for (int k = 0; k < split.frames; ++k) {
    if (std::ranges::contains(summary.inputs, frame_input_name(split.input, k), &OnnxInput::name))
      return refuse(RefusalCode::FormatUnsupported,
                    std::format("ONNX : une entrée s'appelle déjà « {} »", frame_input_name(split.input, k)));
  }
  return {};
}

}  // namespace

std::string frame_input_name(std::string_view input, int k) {
  return std::format("{}_{}", input, k);
}

std::expected<std::vector<std::byte>, Refusal> with_frame_inputs(std::span<const std::byte> model,
                                                                 const FrameSplit& split) {
  auto summary = scan_onnx(model);
  if (!summary) return std::unexpected(std::move(summary.error()));
  const OnnxInput* found = nullptr;
  if (auto ok = check_split(*summary, split, found); !ok) return std::unexpected(std::move(ok.error()));
  std::vector<std::int64_t> dims = found->dims;
  dims[1] /= split.frames;
  const std::int32_t elem_type = found->elem_type;

  auto graph = onnx_graph_field(model);
  if (!graph) return std::unexpected(std::move(graph.error()));
  // Les nœuds ajoutés d'abord : les nœuds restent dans l'ordre d'un tri topologique.
  Message new_graph;
  head_nodes(new_graph, split);
  bool output_found = false;
  bool renamed = false;
  if (auto ok = copy_graph(*graph, split, new_graph, output_found, renamed); !ok)
    return std::unexpected(std::move(ok.error()));
  if (split.half && (!output_found || !renamed))
    return refuse(RefusalCode::FieldMissing, std::format("ONNX : pas de sortie « {} »", split.output));
  if (split.half) tail_nodes(new_graph, split);
  for (int k = 0; k < split.frames; ++k)
    new_graph.message(kGraphInput, value_info(frame_input_name(split.input, k), elem_type, dims));

  std::vector<std::byte> out;
  out.reserve(model.size() + 1024);
  Fields model_fields(model);
  RawField field;
  while (model_fields.next(field)) {
    if (field.number == kModelGraph && field.delimited) {
      protobuf::put_delimited(out, kModelGraph, new_graph.data());
    } else {
      out.insert(out.end(), field.raw.begin(), field.raw.end());
    }
  }
  if (!model_fields.ok()) return refuse(RefusalCode::ModelUnreadable, "ONNX : protobuf du modèle malformé");
  return out;
}

}  // namespace kirari::prepare
