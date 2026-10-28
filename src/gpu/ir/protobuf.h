// Lecture du format filaire protobuf : varint, longueur, fixed32 et fixed64.
//
// Juste ce qu'il faut pour décoder un ModelProto ONNX sans dépendance. Chaque
// lecture vérifie ses bornes : un message tronqué ou incohérent devient un
// refus OnnxMalformed, jamais une lecture hors du tampon. Les groupes
// (types filaires 3 et 4), abandonnés par protobuf, sont refusés.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/types.h"

namespace ajp::gpu::ir::proto {

using Bytes = std::span<const std::byte>;

enum class WireType : std::uint8_t { Varint = 0, Fixed64 = 1, Length = 2, Fixed32 = 5 };

// Un champ décodé. scalar porte la valeur d'un Varint, Fixed64 ou Fixed32 ;
// bytes porte le contenu d'un champ Length, sous-vue du tampon d'origine.
struct Field {
  std::uint32_t number = 0;
  WireType wire = WireType::Varint;
  std::uint64_t scalar = 0;
  Bytes bytes;
};

class Reader {
 public:
  explicit Reader(Bytes data) : data_(data) {}

  // Champ suivant ; std::nullopt à la fin exacte du message.
  std::expected<std::optional<Field>, Refusal> next();

 private:
  std::expected<std::uint64_t, Refusal> varint();
  std::expected<Bytes, Refusal> take(std::size_t count);

  Bytes data_;
  std::size_t pos_ = 0;
};

// Écriture du format filaire, pour recopier un message champ par champ en n'en changeant que
// quelques-uns : un champ lu se réécrit tel quel (field), les autres se posent par leur numéro.
class Writer {
 public:
  void varint(std::uint32_t number, std::uint64_t value);
  void fixed32(std::uint32_t number, std::uint32_t value);
  void fixed64(std::uint32_t number, std::uint64_t value);
  void bytes(std::uint32_t number, Bytes data);
  void string(std::uint32_t number, std::string_view text);
  void message(std::uint32_t number, const Writer& inner) { bytes(number, inner.data()); }
  // Tel qu'il a été lu ; un varint non canonique est réécrit au plus court.
  void field(const Field& field);

  Bytes data() const { return out_; }
  std::vector<std::byte> take() && { return std::move(out_); }

 private:
  void tag(std::uint32_t number, WireType wire);
  void raw_varint(std::uint64_t value);
  void little_endian(std::uint64_t value, int width);

  std::vector<std::byte> out_;
};

std::unexpected<Refusal> malformed(std::string detail);

// Conversions d'un champ vers les types protobuf, avec vérification du type filaire.
std::expected<std::int64_t, Refusal> as_int64(const Field& field);
std::expected<std::int32_t, Refusal> as_int32(const Field& field);  // refuse hors de int32
std::expected<float, Refusal> as_float(const Field& field);
std::expected<std::string, Refusal> as_string(const Field& field);

// Champs répétés : protobuf accepte la forme empaquetée (Length) et la forme
// élément par élément ; les deux s'ajoutent à out.
std::expected<void, Refusal> append_int64s(const Field& field, std::vector<std::int64_t>& out);
std::expected<void, Refusal> append_floats(const Field& field, std::vector<float>& out);

}  // namespace ajp::gpu::ir::proto
