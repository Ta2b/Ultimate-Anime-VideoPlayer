// Décodeur ONNX, moitié tenseurs et attributs : TensorProto (données brutes ou typées, FLOAT, FLOAT16, INT32,
// INT64, FLOAT8E4M3FN) et AttributeProto. Le graphe, les nœuds et les entrées sont dans onnx_reader.cpp.
#include <bit>
#include <format>
#include <limits>
#include <optional>
#include <utility>

#include "gpu/ir/fp8.h"
#include "gpu/ir/graph.h"
#include "gpu/ir/onnx_model.h"
#include "gpu/ir/onnx_reader_detail.h"
#include "gpu/ir/protobuf.h"
#include "kernels/ref/half.h"

namespace ajp::gpu::ir::onnx {
namespace detail {
namespace {

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
    case kFloat8E4M3Fn:
      return 1;
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
  if (tensor.data_type == kFloat || tensor.data_type == kFloat16 || tensor.data_type == kFloat8E4M3Fn) {
    tensor.floats.reserve(count);
  } else {
    tensor.ints.reserve(count);
  }
  for (std::size_t i = 0; i < count; ++i) {
    switch (tensor.data_type) {
      case kFloat:
        tensor.floats.push_back(std::bit_cast<float>(proto::read_le32(raw.subspan(4 * i, 4))));
        break;
      case kFloat16: {
        const auto lo = std::to_integer<std::uint16_t>(raw[2 * i]);
        const auto hi = std::to_integer<std::uint16_t>(raw[(2 * i) + 1]);
        tensor.floats.push_back(kernels::ref::half_to_float(static_cast<std::uint16_t>(lo | (hi << 8U))));
        break;
      }
      case kFloat8E4M3Fn:
        tensor.floats.push_back(e4m3_to_float(std::to_integer<std::uint8_t>(raw[i])));
        break;
      case kInt32:
        tensor.ints.push_back(static_cast<std::int32_t>(proto::read_le32(raw.subspan(4 * i, 4))));
        break;
      default:
        tensor.ints.push_back(static_cast<std::int64_t>(proto::read_le64(raw.subspan(8 * i, 8))));
        break;
    }
  }
  return {};
}

// FLOAT8E4M3FN en int32_data : un octet par élément, dans les 8 bits de poids faible.
std::expected<void, Refusal> decode_e4m3_words(Tensor& tensor, const std::vector<std::int64_t>& words) {
  for (const std::int64_t word : words) {
    if (word < 0 || word > 0xFF) {
      return proto::malformed(
          std::format("tenseur {} : motif FLOAT8E4M3FN {} hors de 8 bits", tensor.name, word));
    }
    tensor.floats.push_back(e4m3_to_float(static_cast<std::uint8_t>(word)));
  }
  return {};
}

// INT32 en int32_data : des varints de 64 bits, à refuser hors de la plage plutôt qu'à tronquer plus loin
// (A44-07).
std::expected<void, Refusal> check_int32_words(const Tensor& tensor, const std::vector<std::int64_t>& words) {
  for (const std::int64_t word : words) {
    if (word < std::numeric_limits<std::int32_t>::min() || word > std::numeric_limits<std::int32_t>::max()) {
      return proto::malformed(std::format("tenseur {} : valeur INT32 {} hors plage", tensor.name, word));
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

// int32_data selon le type du tenseur : motifs FLOAT16 ou FLOAT8E4M3FN, ou valeurs INT32.
std::expected<void, Refusal> decode_int32_data(Tensor& tensor, std::vector<std::int64_t>& words) {
  if (tensor.data_type == kFloat16) return decode_half_words(tensor, words);
  if (tensor.data_type == kFloat8E4M3Fn) return decode_e4m3_words(tensor, words);
  if (tensor.data_type != kInt32) return {};
  if (auto ok = check_int32_words(tensor, words); !ok) return ok;
  tensor.ints = std::move(words);
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
    default:  // kFloat16, kFloat8E4M3Fn, kInt32
      placed = !has_floats && !has_int64;
      break;
  }
  if (!placed)
    return proto::malformed(std::format("tenseur {} : valeurs dans le champ d'un autre type", tensor.name));

  if (auto ok = decode_int32_data(tensor, raw.int32_data); !ok) return std::unexpected(ok.error());
  const std::size_t present = tensor.floats.size() + tensor.ints.size();
  if (present != count) {
    return proto::malformed(
        std::format("tenseur {} : {} valeurs typées pour {} éléments", tensor.name, present, count));
  }
  return std::move(tensor);
}

// C44-03 : les dimensions d'un tenseur ont le même plafond que celles d'une forme (kMaxRank). Un entier
// empaqueté tient en dix octets au plus : un champ plus long en porte donc plus de kMaxRank, refusé avant
// d'être lu.
std::expected<void, Refusal> read_dims(const Field& field, std::vector<std::int64_t>& dims) {
  constexpr std::size_t kMaxPackedBytes = kMaxRank * 10;
  const bool packed = field.wire == proto::WireType::Length;
  if ((packed && field.bytes.size() > kMaxPackedBytes) || (!packed && dims.size() >= kMaxRank)) {
    return proto::malformed("tenseur : trop de valeurs dans dims");
  }
  if (auto ok = proto::append_int64s(field, dims); !ok) return ok;
  if (dims.size() > kMaxRank) return proto::malformed("tenseur : trop de valeurs dans dims");
  return {};
}

std::expected<Tensor, Refusal> read_tensor(Bytes bytes) {
  RawTensor raw;
  auto ok = for_each_field(bytes, [&](const Field& field) -> std::expected<void, Refusal> {
    switch (field.number) {
      case 1:
        return read_dims(field, raw.tensor.dims);
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
        return append_ints(field, raw.int32_data);
      case 7:
        return append_ints(field, raw.tensor.ints);
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

}  // namespace

std::expected<void, Refusal> push_tensor(const Field& field, std::vector<Tensor>& out) {
  if (field.wire != proto::WireType::Length) return proto::malformed("tenseur : message attendu");
  auto tensor = read_tensor(field.bytes);
  if (!tensor) return std::unexpected(tensor.error());
  out.push_back(std::move(*tensor));
  return {};
}

namespace {

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
    case 5:  // t : un seul tenseur, le dernier gagne (Constant ne lit que `tensors.size() == 1`)
      attr.tensors.clear();
      return push_tensor(field, attr.tensors);
    case 10:  // tensors : aucun opérateur de l'IR ne le lit, et chaque tenseur vide coûte une centaine
              // d'octets (C44-04)
      return {};
    case 7:
      return proto::append_floats(field, attr.floats);
    case 8:
      return append_ints(field, attr.ints);
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

}  // namespace

std::expected<Attribute, Refusal> read_attribute(Bytes bytes) {
  Attribute attr;
  auto ok = for_each_field(bytes, [&](const Field& field) { return read_attribute_field(field, attr); });
  if (!ok) return std::unexpected(ok.error());
  return attr;
}

}  // namespace detail
}  // namespace ajp::gpu::ir::onnx
