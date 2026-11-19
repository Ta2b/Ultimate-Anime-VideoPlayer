// Écrire des morceaux de graphe ONNX au niveau du format protobuf, sans dépendance onnx : nœuds, attributs,
// descriptions d'entrées. Partagé par les découpes qui ajoutent des nœuds à un graphe (onnx_frames,
// onnx_cache). Numéros de champ d'onnx.proto (ONNX 1.19.1) : NodeProto.input = 1, .output = 2, .name = 3,
// .op_type = 4, .attribute = 5 ; AttributeProto.name = 1, .i = 3, .s = 4, .ints = 8, .type = 20 (INT = 2,
// STRING = 3, INTS = 7) ; ValueInfoProto.name = 1, .type = 2 ; TypeProto.tensor_type = 1 ;
// TypeProto.Tensor.elem_type = 1, .shape = 2 ; TensorShapeProto.dim = 1 ; Dimension.dim_value = 1,
// .dim_param = 2.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "prepare/protobuf_fields.h"

namespace kirari::prepare::onnx_build {

inline constexpr std::uint32_t kNodeInput = 1;
inline constexpr std::uint32_t kNodeOutput = 2;
inline constexpr std::uint32_t kNodeName = 3;
inline constexpr std::uint32_t kNodeOpType = 4;
inline constexpr std::uint32_t kNodeAttribute = 5;
inline constexpr std::uint32_t kValueInfoName = 1;

inline protobuf::Bytes as_bytes(std::string_view text) {
  return {reinterpret_cast<const std::byte*>(text.data()), text.size()};
}

inline std::string_view as_text(protobuf::Bytes bytes) {
  return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

class Message {
 public:
  Message& varint(std::uint32_t number, std::uint64_t value) {
    protobuf::put_varint(out_, std::uint64_t{number} << 3U);
    protobuf::put_varint(out_, value);
    return *this;
  }
  Message& bytes(std::uint32_t number, protobuf::Bytes data) {
    protobuf::put_delimited(out_, number, data);
    return *this;
  }
  Message& text(std::uint32_t number, std::string_view value) { return bytes(number, as_bytes(value)); }
  Message& message(std::uint32_t number, const Message& inner) { return bytes(number, inner.out_); }
  Message& raw(protobuf::Bytes field) {  // un champ recopié tel quel, son en-tête compris
    out_.insert(out_.end(), field.begin(), field.end());
    return *this;
  }
  const std::vector<std::byte>& data() const { return out_; }

 private:
  std::vector<std::byte> out_;
};

// ValueInfoProto { name ; type { tensor_type { elem_type ; shape { dim… } } } } ; une dimension libre (-1)
// prend un nom selon son rang (batch, channels, height, width) : des entrées de mêmes noms partagent ainsi
// leurs dimensions libres.
inline Message value_info(std::string_view name, std::int32_t elem_type, std::span<const std::int64_t> dims) {
  constexpr std::array<std::string_view, 4> kNames = {"batch", "channels", "height", "width"};
  Message shape;
  for (std::size_t i = 0; i < dims.size(); ++i) {
    Message dim;
    if (dims[i] >= 0) {
      dim.varint(1, static_cast<std::uint64_t>(dims[i]));
    } else {
      dim.text(2, i < kNames.size() ? kNames[i] : std::string_view("dim"));
    }
    shape.message(1, dim);
  }
  Message tensor;
  tensor.varint(1, static_cast<std::uint64_t>(static_cast<std::uint32_t>(elem_type))).message(2, shape);
  Message info;
  info.text(kValueInfoName, name).message(2, Message().message(1, tensor));
  return info;
}

inline Message node(std::string_view op, std::string_view name, std::span<const std::string_view> inputs,
                    std::string_view output) {
  Message n;
  for (const std::string_view in : inputs) n.text(kNodeInput, in);
  n.text(kNodeOutput, output).text(kNodeName, name).text(kNodeOpType, op);
  return n;
}

inline Message int_attribute(std::string_view name, std::int64_t value) {
  return std::move(Message().text(1, name).varint(3, static_cast<std::uint64_t>(value)).varint(20, 2));
}

inline Message ints_attribute(std::string_view name, std::span<const std::int64_t> values) {
  Message a;
  a.text(1, name);
  for (const std::int64_t v : values) a.varint(8, static_cast<std::uint64_t>(v));
  a.varint(20, 7);
  return a;
}

inline Message text_attribute(std::string_view name, std::string_view value) {
  return std::move(Message().text(1, name).text(4, value).varint(20, 3));
}

// AveragePool 2 × 2, pas de 2 : la moyenne de chaque bloc, exacte sur une taille paire.
inline Message pool_node(std::string_view name, std::string_view input, std::string_view output) {
  constexpr std::array<std::int64_t, 2> kTwo = {2, 2};
  Message n = node("AveragePool", name, std::array{input}, output);
  n.message(kNodeAttribute, ints_attribute("kernel_shape", kTwo))
      .message(kNodeAttribute, ints_attribute("strides", kTwo));
  return n;
}

}  // namespace kirari::prepare::onnx_build
