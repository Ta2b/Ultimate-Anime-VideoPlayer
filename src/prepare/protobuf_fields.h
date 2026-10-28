// Lecture et écriture minimales du format protobuf, champ par champ, pour modifier un modèle
// ONNX sans dépendance onnx : on ne descend que dans les messages connus, le reste est recopié
// octet pour octet. Interne à prepare/.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

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
        field.scalar = varint();
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
