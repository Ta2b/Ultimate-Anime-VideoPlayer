// Numéros de champ d'onnx.proto (ONNX 1.19.1) : ModelProto.graph = 7 ; GraphProto.node = 1,
// GraphProto.output = 12 ; NodeProto.output = 2 ; ValueInfoProto.name = 1, .type = 2 ;
// TypeProto.tensor_type = 1 ; TypeProto.Tensor.elem_type = 1.
#include "prepare/onnx_outputs.h"

#include <algorithm>
#include <format>
#include <optional>
#include <string_view>
#include <utility>

namespace ajp::prepare {
namespace {

using Bytes = std::span<const std::byte>;

constexpr std::uint32_t kModelGraph = 7;
constexpr std::uint32_t kGraphNode = 1;
constexpr std::uint32_t kGraphOutput = 12;
constexpr std::uint32_t kNodeOutput = 2;

std::unexpected<Refusal> malformed(std::string detail) {
  return std::unexpected(Refusal{RefusalCode::ModelUnreadable, std::move(detail)});
}

// Un champ protobuf : son numéro, son contenu (délimité) et ses octets bruts, étiquette comprise.
struct RawField {
  std::uint32_t number = 0;
  bool delimited = false;
  Bytes payload;
  Bytes raw;
};

class Fields {
 public:
  explicit Fields(Bytes data) : data_(data) {}

  bool next(RawField& field) {
    if (!ok_ || pos_ >= data_.size()) return false;
    const std::size_t start = pos_;
    const std::uint64_t tag = varint();
    if (!ok_ || (tag >> 3U) == 0 || (tag >> 3U) > 0x1FFFFFFFU) return fail();
    field = RawField{.number = static_cast<std::uint32_t>(tag >> 3U)};
    switch (tag & 7U) {
      case 0:
        varint();
        break;
      case 1:
        if (!skip(8)) return false;
        break;
      case 5:
        if (!skip(4)) return false;
        break;
      case 2: {
        const std::uint64_t n = varint();
        if (!ok_ || data_.size() - pos_ < n) return fail();
        field.payload = data_.subspan(pos_, static_cast<std::size_t>(n));
        field.delimited = true;
        pos_ += static_cast<std::size_t>(n);
        break;
      }
      default:
        return fail();
    }
    if (!ok_) return false;
    field.raw = data_.subspan(start, pos_ - start);
    return true;
  }
  bool ok() const { return ok_; }

 private:
  std::uint64_t varint() {
    std::uint64_t value = 0;
    for (unsigned shift = 0; pos_ < data_.size() && shift < 64; shift += 7) {
      const auto byte = std::to_integer<std::uint64_t>(data_[pos_++]);
      value |= (byte & 0x7fU) << shift;
      if ((byte & 0x80U) == 0) return value;
    }
    ok_ = false;
    return 0;
  }
  bool skip(std::size_t n) {
    if (data_.size() - pos_ < n) return fail();
    pos_ += n;
    return true;
  }
  bool fail() {
    ok_ = false;
    return false;
  }

  Bytes data_;
  std::size_t pos_ = 0;
  bool ok_ = true;
};

void put_varint(std::vector<std::byte>& out, std::uint64_t value) {
  while (value >= 0x80U) {
    out.push_back(static_cast<std::byte>((value & 0x7fU) | 0x80U));
    value >>= 7U;
  }
  out.push_back(static_cast<std::byte>(value));
}

void put_delimited(std::vector<std::byte>& out, std::uint32_t number, Bytes payload) {
  put_varint(out, (std::uint64_t{number} << 3U) | 2U);
  put_varint(out, payload.size());
  out.insert(out.end(), payload.begin(), payload.end());
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
