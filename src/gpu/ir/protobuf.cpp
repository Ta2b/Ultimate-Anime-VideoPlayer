#include "gpu/ir/protobuf.h"

#include <bit>
#include <format>
#include <limits>

namespace ajp::gpu::ir::proto {
namespace {

// Un varint de 64 bits tient en 10 octets au plus.
constexpr int kMaxVarintBytes = 10;
// Numéro de champ maximal du format filaire (29 bits).
constexpr std::uint64_t kMaxFieldNumber = (1U << 29U) - 1U;

std::uint32_t read_le32(Bytes bytes) {
  std::uint32_t value = 0;
  for (std::size_t i = 0; i < 4; ++i) {
    value |= static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(bytes[i])) << (8U * i);
  }
  return value;
}

std::uint64_t read_le64(Bytes bytes) {
  std::uint64_t value = 0;
  for (std::size_t i = 0; i < 8; ++i) {
    value |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(bytes[i])) << (8U * i);
  }
  return value;
}

std::expected<void, Refusal> expect_wire(const Field& field, WireType wire, const char* what) {
  if (field.wire != wire) {
    return malformed(std::format("champ {} : type filaire {} au lieu de {} ({})", field.number,
                                 static_cast<int>(field.wire), static_cast<int>(wire), what));
  }
  return {};
}

// Varint à la position pos de data ; pos avance au-delà.
std::expected<std::uint64_t, Refusal> read_varint(Bytes data, std::size_t& pos) {
  std::uint64_t value = 0;
  for (int i = 0; i < kMaxVarintBytes; ++i) {
    if (pos >= data.size()) return malformed(std::format("varint tronqué à l'octet {}", pos));
    const auto byte = std::to_integer<std::uint8_t>(data[pos++]);
    // Le dixième octet ne porte plus qu'un bit utile.
    if (i == kMaxVarintBytes - 1 && byte > 1U) return malformed("varint au-delà de 64 bits");
    value |= static_cast<std::uint64_t>(byte & 0x7FU) << static_cast<unsigned>(7 * i);
    if ((byte & 0x80U) == 0U) return value;
  }
  return malformed("varint de plus de 10 octets");
}

}  // namespace

std::unexpected<Refusal> malformed(std::string detail) {
  return std::unexpected(Refusal{RefusalCode::OnnxMalformed, std::move(detail)});
}

std::expected<std::uint64_t, Refusal> Reader::varint() {
  return read_varint(data_, pos_);
}

std::expected<Bytes, Refusal> Reader::take(std::size_t count) {
  if (count > data_.size() - pos_) {
    return malformed(std::format("{} octets demandés, {} restants", count, data_.size() - pos_));
  }
  Bytes out = data_.subspan(pos_, count);
  pos_ += count;
  return out;
}

std::expected<std::optional<Field>, Refusal> Reader::next() {
  if (pos_ == data_.size()) return std::optional<Field>{};
  auto tag = varint();
  if (!tag) return std::unexpected(tag.error());
  const std::uint64_t number = *tag >> 3U;
  if (number == 0 || number > kMaxFieldNumber) return malformed(std::format("numéro de champ {}", number));

  Field field{.number = static_cast<std::uint32_t>(number)};
  switch (*tag & 7U) {
    case 0: {
      field.wire = WireType::Varint;
      auto value = varint();
      if (!value) return std::unexpected(value.error());
      field.scalar = *value;
      break;
    }
    case 1: {
      field.wire = WireType::Fixed64;
      auto bytes = take(8);
      if (!bytes) return std::unexpected(bytes.error());
      field.scalar = read_le64(*bytes);
      break;
    }
    case 2: {
      field.wire = WireType::Length;
      auto length = varint();
      if (!length) return std::unexpected(length.error());
      if (*length > data_.size() - pos_) {
        return malformed(std::format("champ {} : longueur {} au-delà du message", number, *length));
      }
      auto bytes = take(static_cast<std::size_t>(*length));
      if (!bytes) return std::unexpected(bytes.error());
      field.bytes = *bytes;
      break;
    }
    case 5: {
      field.wire = WireType::Fixed32;
      auto bytes = take(4);
      if (!bytes) return std::unexpected(bytes.error());
      field.scalar = read_le32(*bytes);
      break;
    }
    default:
      return malformed(std::format("champ {} : type filaire {} non pris en charge", number, *tag & 7U));
  }
  return field;
}

std::expected<std::int64_t, Refusal> as_int64(const Field& field) {
  if (auto ok = expect_wire(field, WireType::Varint, "entier"); !ok) return std::unexpected(ok.error());
  // int64 protobuf : complément à deux sur 64 bits, sans zigzag.
  return static_cast<std::int64_t>(field.scalar);
}

std::expected<std::int32_t, Refusal> as_int32(const Field& field) {
  auto value = as_int64(field);
  if (!value) return std::unexpected(value.error());
  if (*value < std::numeric_limits<std::int32_t>::min() ||
      *value > std::numeric_limits<std::int32_t>::max()) {
    return malformed(std::format("champ {} : {} hors de int32", field.number, *value));
  }
  return static_cast<std::int32_t>(*value);
}

std::expected<float, Refusal> as_float(const Field& field) {
  if (auto ok = expect_wire(field, WireType::Fixed32, "float"); !ok) return std::unexpected(ok.error());
  return std::bit_cast<float>(static_cast<std::uint32_t>(field.scalar));
}

std::expected<std::string, Refusal> as_string(const Field& field) {
  if (auto ok = expect_wire(field, WireType::Length, "chaîne"); !ok) return std::unexpected(ok.error());
  std::string out(field.bytes.size(), '\0');
  for (std::size_t i = 0; i < field.bytes.size(); ++i) {
    out[i] = static_cast<char>(std::to_integer<unsigned char>(field.bytes[i]));
  }
  return out;
}

std::expected<void, Refusal> append_int64s(const Field& field, std::vector<std::int64_t>& out) {
  if (field.wire == WireType::Varint) {
    out.push_back(static_cast<std::int64_t>(field.scalar));
    return {};
  }
  if (auto ok = expect_wire(field, WireType::Length, "entiers empaquetés"); !ok) {
    return std::unexpected(ok.error());
  }
  // Une liste empaquetée est une suite de varints sans étiquette.
  std::size_t pos = 0;
  while (pos < field.bytes.size()) {
    auto value = read_varint(field.bytes, pos);
    if (!value) return std::unexpected(value.error());
    out.push_back(static_cast<std::int64_t>(*value));
  }
  return {};
}

std::expected<void, Refusal> append_floats(const Field& field, std::vector<float>& out) {
  if (field.wire == WireType::Fixed32) {
    out.push_back(std::bit_cast<float>(static_cast<std::uint32_t>(field.scalar)));
    return {};
  }
  if (auto ok = expect_wire(field, WireType::Length, "floats empaquetés"); !ok) {
    return std::unexpected(ok.error());
  }
  if (field.bytes.size() % 4 != 0)
    return malformed("liste de floats empaquetés d'une taille non multiple de 4");
  out.reserve(out.size() + field.bytes.size() / 4);
  for (std::size_t pos = 0; pos < field.bytes.size(); pos += 4) {
    out.push_back(std::bit_cast<float>(read_le32(field.bytes.subspan(pos, 4))));
  }
  return {};
}

}  // namespace ajp::gpu::ir::proto
