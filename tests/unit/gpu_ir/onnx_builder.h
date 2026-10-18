// Encodeur protobuf et constructeur de modèles ONNX, pour les tests seulement.
//
// Écrit indépendamment du décodeur testé : chaque champ est posé à la main
// d'après onnx.proto, octet par octet. Aucun fichier binaire dans le dépôt :
// les modèles des tests naissent ici.
#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gpu/ir/half.h"
#include "gpu/ir/onnx_model.h"

namespace ajp::gpu::ir::test {

class ProtoWriter {
 public:
  // Varint brut, sans étiquette.
  ProtoWriter& raw_varint(std::uint64_t value) {
    while (value >= 0x80) {
      out_.push_back(static_cast<char>((value & 0x7F) | 0x80));
      value >>= 7;
    }
    out_.push_back(static_cast<char>(value));
    return *this;
  }

  ProtoWriter& tag(std::uint32_t field, std::uint32_t wire) {
    return raw_varint((static_cast<std::uint64_t>(field) << 3) | wire);
  }

  // int64 et int32 protobuf : complément à deux sur 64 bits, sans zigzag.
  ProtoWriter& varint(std::uint32_t field, std::int64_t value) {
    return tag(field, 0).raw_varint(static_cast<std::uint64_t>(value));
  }

  ProtoWriter& fixed32(std::uint32_t field, std::uint32_t bits) {
    tag(field, 5);
    for (int k = 0; k < 4; ++k) out_.push_back(static_cast<char>((bits >> (8 * k)) & 0xFF));
    return *this;
  }

  ProtoWriter& fixed64(std::uint32_t field, std::uint64_t bits) {
    tag(field, 1);
    for (int k = 0; k < 8; ++k) out_.push_back(static_cast<char>((bits >> (8 * k)) & 0xFF));
    return *this;
  }

  ProtoWriter& f32(std::uint32_t field, float value) {
    return fixed32(field, std::bit_cast<std::uint32_t>(value));
  }

  ProtoWriter& bytes(std::uint32_t field, std::string_view data) {
    tag(field, 2).raw_varint(data.size());
    out_.append(data);
    return *this;
  }

  ProtoWriter& message(std::uint32_t field, const ProtoWriter& inner) { return bytes(field, inner.str()); }

  ProtoWriter& packed_varints(std::uint32_t field, std::span<const std::int64_t> values) {
    ProtoWriter inner;
    for (const std::int64_t v : values) inner.raw_varint(static_cast<std::uint64_t>(v));
    return bytes(field, inner.str());
  }

  ProtoWriter& packed_floats(std::uint32_t field, std::span<const float> values) {
    std::string data;
    for (const float v : values) append_le(data, std::bit_cast<std::uint32_t>(v), 4);
    return bytes(field, data);
  }

  ProtoWriter& append(std::string_view data) {
    out_.append(data);
    return *this;
  }

  const std::string& str() const { return out_; }

  std::vector<std::byte> data() const {
    std::vector<std::byte> out(out_.size());
    for (std::size_t k = 0; k < out_.size(); ++k) out[k] = static_cast<std::byte>(out_[k]);
    return out;
  }

  static void append_le(std::string& data, std::uint64_t bits, int width) {
    for (int k = 0; k < width; ++k) data.push_back(static_cast<char>((bits >> (8 * k)) & 0xFF));
  }

 private:
  std::string out_;
};

// TensorProto : dims (1), data_type (2), name (8), puis les données.
inline ProtoWriter tensor_header(std::string_view name, std::int32_t type,
                                 std::span<const std::int64_t> dims) {
  ProtoWriter t;
  for (const std::int64_t d : dims) t.varint(1, d);
  t.varint(2, type);
  t.bytes(8, name);
  return t;
}

// FLOAT en raw_data (9), petit-boutiste.
inline ProtoWriter tensor_f32(std::string_view name, const std::vector<std::int64_t>& dims,
                              std::span<const float> values) {
  std::string raw;
  for (const float v : values) ProtoWriter::append_le(raw, std::bit_cast<std::uint32_t>(v), 4);
  return tensor_header(name, onnx::kFloat, dims).bytes(9, raw);
}

// FLOAT16 en raw_data : les valeurs sont arrondies au F16 le plus proche.
inline ProtoWriter tensor_f16(std::string_view name, const std::vector<std::int64_t>& dims,
                              std::span<const float> values) {
  std::string raw;
  for (const float v : values) ProtoWriter::append_le(raw, float_to_half(v), 2);
  return tensor_header(name, onnx::kFloat16, dims).bytes(9, raw);
}

// INT64 en int64_data (7), empaqueté.
inline ProtoWriter tensor_i64(std::string_view name, const std::vector<std::int64_t>& dims,
                              std::span<const std::int64_t> values) {
  return tensor_header(name, onnx::kInt64, dims).packed_varints(7, values);
}

// AttributeProto : name (1), type (20), puis la valeur.
inline ProtoWriter attr_int(std::string_view name, std::int64_t value) {
  return ProtoWriter{}.bytes(1, name).varint(3, value).varint(20, 2);
}

inline ProtoWriter attr_float(std::string_view name, float value) {
  return ProtoWriter{}.bytes(1, name).f32(2, value).varint(20, 1);
}

inline ProtoWriter attr_string(std::string_view name, std::string_view value) {
  return ProtoWriter{}.bytes(1, name).bytes(4, value).varint(20, 3);
}

inline ProtoWriter attr_ints(std::string_view name, std::vector<std::int64_t> values) {
  return ProtoWriter{}.bytes(1, name).packed_varints(8, values).varint(20, 7);
}

inline ProtoWriter attr_floats(std::string_view name, std::vector<float> values) {
  return ProtoWriter{}.bytes(1, name).packed_floats(7, values).varint(20, 6);
}

inline ProtoWriter attr_tensor(std::string_view name, const ProtoWriter& tensor) {
  return ProtoWriter{}.bytes(1, name).message(5, tensor).varint(20, 4);
}

// ValueInfoProto : name (1), type (2) -> tensor_type (1) -> elem_type (1), shape (2) -> dim (1).
// Une dimension négative devient un dim_param nommé.
inline ProtoWriter value_info(std::string_view name, std::int32_t elem_type,
                              const std::vector<std::int64_t>& dims, bool with_shape = true) {
  ProtoWriter tensor_type;
  tensor_type.varint(1, elem_type);
  if (with_shape) {
    ProtoWriter shape;
    for (std::size_t k = 0; k < dims.size(); ++k) {
      ProtoWriter dim;
      if (dims[k] >= 0) {
        dim.varint(1, dims[k]);
      } else {
        dim.bytes(2, "d" + std::to_string(k));
      }
      shape.message(1, dim);
    }
    tensor_type.message(2, shape);
  }
  return ProtoWriter{}.bytes(1, name).message(2, ProtoWriter{}.message(1, tensor_type));
}

class ModelBuilder {
 public:
  explicit ModelBuilder(std::int64_t opset = 17) : opset_(opset) {}

  ModelBuilder& input(std::string_view name, const std::vector<std::int64_t>& dims,
                      std::int32_t type = onnx::kFloat) {
    graph_.message(11, value_info(name, type, dims));
    return *this;
  }

  ModelBuilder& output(std::string_view name, std::int32_t type = onnx::kFloat) {
    graph_.message(12, value_info(name, type, {}, false));
    return *this;
  }

  ModelBuilder& initializer(const ProtoWriter& tensor) {
    graph_.message(5, tensor);
    return *this;
  }

  ModelBuilder& weight(std::string_view name, const std::vector<std::int64_t>& dims,
                       const std::vector<float>& values) {
    return initializer(tensor_f32(name, dims, values));
  }

  ModelBuilder& ints(std::string_view name, const std::vector<std::int64_t>& values) {
    return initializer(tensor_i64(name, {static_cast<std::int64_t>(values.size())}, values));
  }

  // NodeProto : input (1), output (2), name (3), op_type (4), attribute (5), domain (7).
  ModelBuilder& node(std::string_view op_type, std::initializer_list<std::string_view> inputs,
                     std::initializer_list<std::string_view> outputs,
                     std::initializer_list<ProtoWriter> attrs = {}, std::string_view domain = {}) {
    ProtoWriter n;
    for (const std::string_view in : inputs) n.bytes(1, in);
    for (const std::string_view out : outputs) n.bytes(2, out);
    n.bytes(3, std::string(op_type) + "_" + std::to_string(++nodes_));
    n.bytes(4, op_type);
    for (const ProtoWriter& a : attrs) n.message(5, a);
    if (!domain.empty()) n.bytes(7, domain);
    graph_.message(1, n);
    return *this;
  }

  // Champs de GraphProto écrits tels quels, pour les cas limites.
  ProtoWriter& graph() { return graph_; }

  // ModelProto : ir_version (1), producer_name (2), graph (7), opset_import (8).
  ProtoWriter model() const {
    ProtoWriter m;
    m.varint(1, 8);
    m.bytes(2, "ajp-test");
    m.message(7, graph_);
    m.message(8, ProtoWriter{}.bytes(1, "").varint(2, opset_));
    return m;
  }

  std::vector<std::byte> bytes() const { return model().data(); }

 private:
  std::int64_t opset_;
  ProtoWriter graph_;
  int nodes_ = 0;
};

}  // namespace ajp::gpu::ir::test
