// Lecture et écriture du format protobuf, champ par champ, pour modifier un modèle ONNX sans
// dépendance onnx : on ne descend que dans les messages connus, le reste est recopié octet pour
// octet. Interne à prepare/ ; la lecture est celle de gpu/ir/protobuf.h.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "gpu/ir/protobuf.h"

namespace ajp::prepare::protobuf {

using Bytes = std::span<const std::byte>;

// Un champ protobuf : son numéro, son contenu (délimité) et ses octets bruts, étiquette comprise.
struct RawField {
  std::uint32_t number = 0;
  bool delimited = false;
  std::uint64_t scalar = 0;  // valeur d'un champ varint
  Bytes payload;
  Bytes raw;
};

// Le lecteur de gpu/ir/protobuf.h (le seul du dépôt, P1.31) sous la forme qu'attend prepare/ :
// next() faux à la fin du message comme sur une erreur, que ok() distingue.
class Fields {
 public:
  explicit Fields(Bytes data) : reader_(data) {}

  bool next(RawField& field) {
    if (!ok_) return false;
    auto read = reader_.next();
    if (!read) {
      ok_ = false;
      return false;
    }
    if (!*read) return false;
    const gpu::ir::proto::Field& got = **read;
    const bool delimited = got.wire == gpu::ir::proto::WireType::Length;
    field = RawField{.number = got.number,
                     .delimited = delimited,
                     .scalar = got.wire == gpu::ir::proto::WireType::Varint ? got.scalar : 0,
                     .payload = delimited ? got.bytes : Bytes{},
                     .raw = got.raw};
    return true;
  }
  bool ok() const { return ok_; }

 private:
  gpu::ir::proto::Reader reader_;
  bool ok_ = true;
};

inline void put_varint(std::vector<std::byte>& out, std::uint64_t value) {
  while (value >= 0x80U) {
    out.push_back(static_cast<std::byte>((value & 0x7fU) | 0x80U));
    value >>= 7U;
  }
  out.push_back(static_cast<std::byte>(value));
}

inline void put_delimited(std::vector<std::byte>& out, std::uint32_t number, Bytes payload) {
  put_varint(out, (std::uint64_t{number} << 3U) | 2U);
  put_varint(out, payload.size());
  out.insert(out.end(), payload.begin(), payload.end());
}

}  // namespace ajp::prepare::protobuf
