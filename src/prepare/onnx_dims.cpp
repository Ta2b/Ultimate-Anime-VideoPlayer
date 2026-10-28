// Numéros de champ d'onnx.proto (ONNX 1.19.1) : ModelProto.graph = 7 ; GraphProto.input = 11,
// .output = 12, .value_info = 13 ; ValueInfoProto.name = 1, .type = 2 ; TypeProto.tensor_type = 1 ;
// TypeProto.Tensor.shape = 2 ; TensorShapeProto.dim = 1 ; Dimension.dim_value = 1, .dim_param = 2.
#include "prepare/onnx_dims.h"

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
constexpr std::uint32_t kGraphInput = 11;
constexpr std::uint32_t kGraphOutput = 12;
constexpr std::uint32_t kGraphValueInfo = 13;
constexpr std::uint32_t kValueInfoName = 1;
constexpr std::uint32_t kValueInfoType = 2;
constexpr std::uint32_t kTypeTensor = 1;
constexpr std::uint32_t kTensorShape = 2;
constexpr std::uint32_t kShapeDim = 1;
constexpr std::uint32_t kDimValue = 1;
constexpr std::uint32_t kDimParam = 2;

std::unexpected<Refusal> malformed(std::string detail) {
  return std::unexpected(Refusal{RefusalCode::ModelUnreadable, std::move(detail)});
}

std::string_view text(Bytes bytes) {
  return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

// Ce que la réécriture d'un message a rencontré : une dimension nommée restée sans valeur.
struct Rewrite {
  std::span<const NamedDim> dims;
  std::optional<std::string> unfixed;
};

using Rebuild = std::expected<std::vector<std::byte>, Refusal> (*)(Bytes, Rewrite&);

// Le message avec chaque champ number réécrit par rebuild, les autres recopiés tels quels.
std::expected<std::vector<std::byte>, Refusal> map_field(Bytes message, std::uint32_t number, Rebuild rebuild,
                                                         Rewrite& rewrite, std::string_view what) {
  std::vector<std::byte> out;
  out.reserve(message.size());
  Fields fields(message);
  RawField field;
  while (fields.next(field)) {
    if (field.number == number && field.delimited) {
      auto inner = rebuild(field.payload, rewrite);
      if (!inner) return std::unexpected(std::move(inner.error()));
      put_delimited(out, number, *inner);
    } else {
      out.insert(out.end(), field.raw.begin(), field.raw.end());
    }
  }
  if (!fields.ok()) return malformed(std::format("ONNX : {} malformé", what));
  return out;
}

std::expected<std::vector<std::byte>, Refusal> dimension(Bytes message, Rewrite& rewrite) {
  Fields fields(message);
  RawField field;
  std::optional<std::string_view> param;
  while (fields.next(field)) {
    if (field.number == kDimParam && field.delimited) param = text(field.payload);
  }
  if (!fields.ok()) return malformed("ONNX : dimension malformée");
  if (!param) return std::vector<std::byte>(message.begin(), message.end());
  const auto known = std::ranges::find(rewrite.dims, *param, &NamedDim::name);
  if (known == rewrite.dims.end()) {
    if (!rewrite.unfixed) rewrite.unfixed = std::string(*param);
    return std::vector<std::byte>(message.begin(), message.end());
  }
  std::vector<std::byte> out;
  put_varint(out, (std::uint64_t{kDimValue} << 3U) | 0U);
  put_varint(out, static_cast<std::uint64_t>(known->value));
  return out;
}

std::expected<std::vector<std::byte>, Refusal> shape(Bytes message, Rewrite& rewrite) {
  return map_field(message, kShapeDim, dimension, rewrite, "forme");
}

std::expected<std::vector<std::byte>, Refusal> tensor_type(Bytes message, Rewrite& rewrite) {
  return map_field(message, kTensorShape, shape, rewrite, "type de tenseur");
}

std::expected<std::vector<std::byte>, Refusal> type(Bytes message, Rewrite& rewrite) {
  return map_field(message, kTypeTensor, tensor_type, rewrite, "type");
}

std::expected<std::vector<std::byte>, Refusal> value_info(Bytes message, Rewrite& rewrite) {
  return map_field(message, kValueInfoType, type, rewrite, "value_info");
}

// Le nom d'un ValueInfoProto, pour le refus.
std::string value_name(Bytes message) {
  Fields fields(message);
  RawField field;
  while (fields.next(field)) {
    if (field.number == kValueInfoName && field.delimited) return std::string(text(field.payload));
  }
  return "?";
}

}  // namespace

std::expected<std::vector<std::byte>, Refusal> with_fixed_dims(std::span<const std::byte> model,
                                                               std::span<const NamedDim> dims) {
  for (const NamedDim& d : dims) {
    if (d.value <= 0) {
      return std::unexpected(
          Refusal{RefusalCode::FieldOutOfRange,
                  std::format("ONNX : dimension « {} » = {}, positive attendue", d.name, d.value)});
    }
  }
  std::optional<Bytes> graph;
  {
    Fields fields(model);
    RawField field;
    while (fields.next(field)) {
      if (field.number == kModelGraph && field.delimited) graph = field.payload;
    }
    if (!fields.ok()) return malformed("ONNX : protobuf du modèle malformé");
  }
  if (!graph) return malformed("ONNX : pas de graphe principal");

  std::vector<std::byte> new_graph;
  new_graph.reserve(graph->size());
  Fields fields(*graph);
  RawField field;
  while (fields.next(field)) {
    const bool io = field.number == kGraphInput || field.number == kGraphOutput;
    if (!field.delimited || (!io && field.number != kGraphValueInfo)) {
      new_graph.insert(new_graph.end(), field.raw.begin(), field.raw.end());
      continue;
    }
    Rewrite rewrite{.dims = dims};
    auto rebuilt = value_info(field.payload, rewrite);
    if (!rebuilt) return std::unexpected(std::move(rebuilt.error()));
    if (io && rewrite.unfixed) {
      return std::unexpected(
          Refusal{RefusalCode::FieldMissing,
                  std::format("ONNX : « {} » garde la dimension « {} », sans valeur donnée",
                              value_name(field.payload), *rewrite.unfixed)});
    }
    put_delimited(new_graph, field.number, *rebuilt);
  }
  if (!fields.ok()) return malformed("ONNX : protobuf du graphe malformé");

  std::vector<std::byte> out;
  out.reserve(model.size());
  Fields model_fields(model);
  while (model_fields.next(field)) {
    if (field.number == kModelGraph && field.delimited) {
      put_delimited(out, kModelGraph, new_graph);
    } else {
      out.insert(out.end(), field.raw.begin(), field.raw.end());
    }
  }
  return out;
}

}  // namespace ajp::prepare
