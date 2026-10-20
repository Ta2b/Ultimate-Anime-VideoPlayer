// Décodeur ONNX sans dépendance, au niveau du format filaire protobuf.
//
// Numéros de champ d'onnx.proto (ONNX 1.17). Seuls les champs déclarés comme
// messages sont relus en protobuf : les données brutes et les chaînes ne le sont
// jamais. Approche voisine de animejanai-inference-rtx@118c272 src/onnx_shape.cpp,
// réécrite pour décoder tout le graphe et refuser au lieu d'ignorer.
#include <bit>
#include <format>
#include <limits>
#include <optional>
#include <utility>

#include "gpu/ir/onnx_model.h"
#include "gpu/ir/protobuf.h"
#include "kernels/ref/half.h"

namespace ajp::gpu::ir::onnx {
namespace {

using proto::Bytes;
using proto::Field;

// Appelle fn sur chaque champ du message ; s'arrête au premier refus.
template <class Fn>
std::expected<void, Refusal> for_each_field(Bytes bytes, Fn&& fn) {
  proto::Reader reader(bytes);
  for (;;) {
    auto field = reader.next();
    if (!field) return std::unexpected(field.error());
    if (!*field) return {};
    if (auto ok = fn(**field); !ok) return ok;
  }
}

std::unexpected<Refusal> unsupported(std::string detail) {
  return std::unexpected(Refusal{RefusalCode::OnnxUnsupported, std::move(detail)});
}

// Lit un champ chaîne dans out.
std::expected<void, Refusal> read_string(const Field& field, std::string& out) {
  auto value = proto::as_string(field);
  if (!value) return std::unexpected(value.error());
  out = std::move(*value);
  return {};
}

std::expected<void, Refusal> push_string(const Field& field, std::vector<std::string>& out) {
  auto value = proto::as_string(field);
  if (!value) return std::unexpected(value.error());
  out.push_back(std::move(*value));
  return {};
}

std::uint32_t le32(Bytes bytes, std::size_t at) {
  std::uint32_t value = 0;
  for (std::size_t i = 0; i < 4; ++i) {
    value |= static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(bytes[at + i])) << (8U * i);
  }
  return value;
}

std::uint64_t le64(Bytes bytes, std::size_t at) {
  return static_cast<std::uint64_t>(le32(bytes, at)) |
         (static_cast<std::uint64_t>(le32(bytes, at + 4)) << 32U);
}

// Ce qu'un TensorProto contient avant validation.
struct RawTensor {
  Tensor tensor;
  std::optional<Bytes> raw;
  std::vector<std::int64_t> int32_data;
  bool external = false;
  bool segmented = false;
};

std::size_t element_size(std::int32_t data_type) {
  switch (data_type) {
    case kFloat:
    case kInt32:
      return 4;
    case kInt64:
      return 8;
    case kFloat16:
      return 2;
    default:
      return 0;
  }
}

std::expected<void, Refusal> decode_raw(Tensor& tensor, Bytes raw, std::size_t count) {
  const std::size_t size = element_size(tensor.data_type);
  if (raw.size() / size != count || raw.size() % size != 0) {
    return proto::malformed(
        std::format("tenseur {} : {} octets bruts pour {} éléments", tensor.name, raw.size(), count));
  }
  if (tensor.data_type == kFloat || tensor.data_type == kFloat16) {
    tensor.floats.reserve(count);
  } else {
    tensor.ints.reserve(count);
  }
  for (std::size_t i = 0; i < count; ++i) {
    switch (tensor.data_type) {
      case kFloat:
        tensor.floats.push_back(std::bit_cast<float>(le32(raw, 4 * i)));
        break;
      case kFloat16: {
        const auto lo = std::to_integer<std::uint16_t>(raw[2 * i]);
        const auto hi = std::to_integer<std::uint16_t>(raw[(2 * i) + 1]);
        tensor.floats.push_back(kernels::ref::half_to_float(static_cast<std::uint16_t>(lo | (hi << 8U))));
        break;
      }
      case kInt32:
        tensor.ints.push_back(static_cast<std::int32_t>(le32(raw, 4 * i)));
        break;
      default:
        tensor.ints.push_back(static_cast<std::int64_t>(le64(raw, 8 * i)));
        break;
    }
  }
  return {};
}

// FLOAT16 en int32_data : un motif binaire de 16 bits par élément.
std::expected<void, Refusal> decode_half_words(Tensor& tensor, const std::vector<std::int64_t>& words) {
  for (const std::int64_t word : words) {
    if (word < 0 || word > 0xFFFF) {
      return proto::malformed(
          std::format("tenseur {} : motif FLOAT16 {} hors de 16 bits", tensor.name, word));
    }
    tensor.floats.push_back(kernels::ref::half_to_float(static_cast<std::uint16_t>(word)));
  }
  return {};
}

std::expected<Tensor, Refusal> finish_tensor(RawTensor raw) {
  Tensor& tensor = raw.tensor;
  if (raw.external) {
    return std::unexpected(Refusal{RefusalCode::OnnxExternalData,
                                   std::format("tenseur {} : poids hors du fichier ONNX", tensor.name)});
  }
  if (raw.segmented) return unsupported(std::format("tenseur {} : segments", tensor.name));
  if (element_size(tensor.data_type) == 0) {
    return unsupported(std::format("tenseur {} : type ONNX {}", tensor.name, tensor.data_type));
  }

  // Nombre d'éléments, borné par la taille du tampon : jamais d'allocation sur une forme seule.
  std::size_t count = 1;
  for (const std::int64_t dim : tensor.dims) {
    if (dim < 0) return proto::malformed(std::format("tenseur {} : dimension {}", tensor.name, dim));
    const auto udim = static_cast<std::uint64_t>(dim);
    if (udim != 0 && count > std::numeric_limits<std::uint32_t>::max() / udim) {
      return proto::malformed(std::format("tenseur {} : trop d'éléments", tensor.name));
    }
    count *= static_cast<std::size_t>(udim);
  }

  const bool typed = !tensor.floats.empty() || !tensor.ints.empty() || !raw.int32_data.empty();
  if (raw.raw && typed) {
    return proto::malformed(std::format("tenseur {} : données brutes et typées à la fois", tensor.name));
  }
  if (raw.raw) {
    if (auto ok = decode_raw(tensor, *raw.raw, count); !ok) return std::unexpected(ok.error());
    return std::move(tensor);
  }

  // Chaque type a son champ typé : float_data, int32_data (INT32 et FLOAT16) ou int64_data.
  const bool has_floats = !tensor.floats.empty();
  const bool has_int32 = !raw.int32_data.empty();
  const bool has_int64 = !tensor.ints.empty();
  bool placed = false;
  switch (tensor.data_type) {
    case kFloat:
      placed = !has_int32 && !has_int64;
      break;
    case kInt64:
      placed = !has_floats && !has_int32;
      break;
    default:  // kFloat16, kInt32
      placed = !has_floats && !has_int64;
      break;
  }
  if (!placed)
    return proto::malformed(std::format("tenseur {} : valeurs dans le champ d'un autre type", tensor.name));

  if (tensor.data_type == kFloat16) {
    if (auto ok = decode_half_words(tensor, raw.int32_data); !ok) return std::unexpected(ok.error());
  } else if (tensor.data_type == kInt32) {
    tensor.ints = std::move(raw.int32_data);
  }
  const std::size_t present = tensor.floats.size() + tensor.ints.size();
  if (present != count) {
    return proto::malformed(
        std::format("tenseur {} : {} valeurs typées pour {} éléments", tensor.name, present, count));
  }
  return std::move(tensor);
}

std::expected<Tensor, Refusal> read_tensor(Bytes bytes) {
  RawTensor raw;
  auto ok = for_each_field(bytes, [&](const Field& field) -> std::expected<void, Refusal> {
    switch (field.number) {
      case 1:
        return proto::append_int64s(field, raw.tensor.dims);
      case 2: {
        auto type = proto::as_int32(field);
        if (!type) return std::unexpected(type.error());
        raw.tensor.data_type = *type;
        return {};
      }
      case 3:
        raw.segmented = true;
        return {};
      case 4:
        return proto::append_floats(field, raw.tensor.floats);
      case 5:
        return proto::append_int64s(field, raw.int32_data);
      case 7:
        return proto::append_int64s(field, raw.tensor.ints);
      case 8:
        return read_string(field, raw.tensor.name);
      case 9:
        if (field.wire != proto::WireType::Length) return proto::malformed("raw_data n'est pas une chaîne");
        raw.raw = field.bytes;
        return {};
      case 13:
        raw.external = true;
        return {};
      case 14: {
        auto location = proto::as_int64(field);
        if (!location) return std::unexpected(location.error());
        raw.external = raw.external || *location == 1;
        return {};
      }
      default:
        return {};
    }
  });
  if (!ok) return std::unexpected(ok.error());
  return finish_tensor(std::move(raw));
}

std::expected<void, Refusal> push_tensor(const Field& field, std::vector<Tensor>& out) {
  if (field.wire != proto::WireType::Length) return proto::malformed("tenseur : message attendu");
  auto tensor = read_tensor(field.bytes);
  if (!tensor) return std::unexpected(tensor.error());
  out.push_back(std::move(*tensor));
  return {};
}

std::expected<void, Refusal> read_attribute_field(const Field& field, Attribute& attr) {
  switch (field.number) {
    case 1:
      return read_string(field, attr.name);
    case 2: {
      auto value = proto::as_float(field);
      if (!value) return std::unexpected(value.error());
      attr.f = *value;
      return {};
    }
    case 3: {
      auto value = proto::as_int64(field);
      if (!value) return std::unexpected(value.error());
      attr.i = *value;
      return {};
    }
    case 4:
      return read_string(field, attr.s);
    case 5:
    case 10:
      return push_tensor(field, attr.tensors);
    case 7:
      return proto::append_floats(field, attr.floats);
    case 8:
      return proto::append_int64s(field, attr.ints);
    case 9:
      return push_string(field, attr.strings);
    case 20: {
      auto type = proto::as_int64(field);
      if (!type) return std::unexpected(type.error());
      if (*type < 0 || *type > static_cast<std::int64_t>(AttributeType::TypeProtos)) {
        return proto::malformed(std::format("attribut {} : type {}", attr.name, *type));
      }
      attr.type = static_cast<AttributeType>(*type);
      return {};
    }
    case 21:
      return unsupported("attribut renvoyant à une fonction (ref_attr_name)");
    default:
      return {};  // graphes, tenseurs creux, doc : sans usage ici
  }
}

std::expected<Attribute, Refusal> read_attribute(Bytes bytes) {
  Attribute attr;
  auto ok = for_each_field(bytes, [&](const Field& field) { return read_attribute_field(field, attr); });
  if (!ok) return std::unexpected(ok.error());
  return attr;
}

std::expected<Node, Refusal> read_node(Bytes bytes) {
  Node node;
  auto ok = for_each_field(bytes, [&](const Field& field) -> std::expected<void, Refusal> {
    switch (field.number) {
      case 1:
        return push_string(field, node.inputs);
      case 2:
        return push_string(field, node.outputs);
      case 3:
        return read_string(field, node.name);
      case 4:
        return read_string(field, node.op_type);
      case 5: {
        if (field.wire != proto::WireType::Length) return proto::malformed("attribut : message attendu");
        auto attr = read_attribute(field.bytes);
        if (!attr) return std::unexpected(attr.error());
        node.attributes.push_back(std::move(*attr));
        return {};
      }
      case 7:
        return read_string(field, node.domain);
      default:
        return {};
    }
  });
  if (!ok) return std::unexpected(ok.error());
  return node;
}

std::expected<void, Refusal> read_dimension(Bytes bytes, ValueInfo& info) {
  std::int64_t value = -1;
  std::string symbol;
  auto ok = for_each_field(bytes, [&](const Field& field) -> std::expected<void, Refusal> {
    if (field.number == 1) {
      auto dim = proto::as_int64(field);
      if (!dim) return std::unexpected(dim.error());
      // Certains exports (RIFE 4.26) écrivent -1 pour une dimension dynamique : elle reste
      // inconnue, comme une dimension sans valeur. Toute autre valeur négative est refusée.
      if (*dim < -1) return proto::malformed(std::format("{} : dimension {}", info.name, *dim));
      value = *dim;
    } else if (field.number == 2) {
      return read_string(field, symbol);
    }
    return {};
  });
  if (!ok) return ok;
  info.dims.push_back(value);
  info.symbols.push_back(value >= 0 ? std::string{} : (symbol.empty() ? std::string{"?"} : symbol));
  return {};
}

std::expected<void, Refusal> read_tensor_type(Bytes bytes, ValueInfo& info) {
  return for_each_field(bytes, [&](const Field& field) -> std::expected<void, Refusal> {
    if (field.number == 1) {
      auto type = proto::as_int32(field);
      if (!type) return std::unexpected(type.error());
      info.elem_type = *type;
    } else if (field.number == 2) {
      if (field.wire != proto::WireType::Length) return proto::malformed("forme : message attendu");
      info.has_shape = true;
      return for_each_field(field.bytes, [&](const Field& dim) -> std::expected<void, Refusal> {
        if (dim.number != 1) return {};
        if (dim.wire != proto::WireType::Length) return proto::malformed("dimension : message attendu");
        return read_dimension(dim.bytes, info);
      });
    }
    return {};
  });
}

std::expected<ValueInfo, Refusal> read_value_info(Bytes bytes) {
  ValueInfo info;
  bool tensor_type = false;
  auto ok = for_each_field(bytes, [&](const Field& field) -> std::expected<void, Refusal> {
    if (field.number == 1) return read_string(field, info.name);
    if (field.number != 2) return {};
    if (field.wire != proto::WireType::Length) return proto::malformed("type : message attendu");
    return for_each_field(field.bytes, [&](const Field& type) -> std::expected<void, Refusal> {
      if (type.number != 1) return {};
      if (type.wire != proto::WireType::Length) return proto::malformed("type tenseur : message attendu");
      tensor_type = true;
      return read_tensor_type(type.bytes, info);
    });
  });
  if (!ok) return std::unexpected(ok.error());
  if (!tensor_type) return unsupported(std::format("{} : seules les entrées et sorties tenseurs", info.name));
  return info;
}

std::expected<void, Refusal> push_value_info(const Field& field, std::vector<ValueInfo>& out) {
  if (field.wire != proto::WireType::Length) return proto::malformed("ValueInfoProto : message attendu");
  auto info = read_value_info(field.bytes);
  if (!info) return std::unexpected(info.error());
  out.push_back(std::move(*info));
  return {};
}

std::expected<Graph, Refusal> read_graph(Bytes bytes) {
  Graph graph;
  auto ok = for_each_field(bytes, [&](const Field& field) -> std::expected<void, Refusal> {
    switch (field.number) {
      case 1: {
        if (field.wire != proto::WireType::Length) return proto::malformed("nœud : message attendu");
        auto node = read_node(field.bytes);
        if (!node) return std::unexpected(node.error());
        graph.nodes.push_back(std::move(*node));
        return {};
      }
      case 2:
        return read_string(field, graph.name);
      case 5:
        return push_tensor(field, graph.initializers);
      case 11:
        return push_value_info(field, graph.inputs);
      case 12:
        return push_value_info(field, graph.outputs);
      case 15:
        return unsupported("initialiseurs creux");
      default:
        return {};
    }
  });
  if (!ok) return std::unexpected(ok.error());
  return graph;
}

std::expected<OpsetId, Refusal> read_opset(Bytes bytes) {
  OpsetId opset;
  auto ok = for_each_field(bytes, [&](const Field& field) -> std::expected<void, Refusal> {
    if (field.number == 1) return read_string(field, opset.domain);
    if (field.number != 2) return {};
    auto version = proto::as_int64(field);
    if (!version) return std::unexpected(version.error());
    opset.version = *version;
    return {};
  });
  if (!ok) return std::unexpected(ok.error());
  return opset;
}

}  // namespace

std::expected<Model, Refusal> read_onnx(std::span<const std::byte> bytes) {
  Model model;
  bool has_graph = false;
  auto ok = for_each_field(bytes, [&](const Field& field) -> std::expected<void, Refusal> {
    switch (field.number) {
      case 1: {
        auto version = proto::as_int64(field);
        if (!version) return std::unexpected(version.error());
        model.ir_version = *version;
        return {};
      }
      case 2:
        return read_string(field, model.producer_name);
      case 3:
        return read_string(field, model.producer_version);
      case 7: {
        if (field.wire != proto::WireType::Length) return proto::malformed("graphe : message attendu");
        if (has_graph) return proto::malformed("plusieurs graphes");
        auto graph = read_graph(field.bytes);
        if (!graph) return std::unexpected(graph.error());
        model.graph = std::move(*graph);
        has_graph = true;
        return {};
      }
      case 8: {
        if (field.wire != proto::WireType::Length) return proto::malformed("opset : message attendu");
        auto opset = read_opset(field.bytes);
        if (!opset) return std::unexpected(opset.error());
        model.opsets.push_back(std::move(*opset));
        return {};
      }
      default:
        return {};
    }
  });
  if (!ok) return std::unexpected(ok.error());
  if (!has_graph) return proto::malformed("ModelProto sans graphe");
  return model;
}

}  // namespace ajp::gpu::ir::onnx
