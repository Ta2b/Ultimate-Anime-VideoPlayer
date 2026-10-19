#include "gpu/ir/protobuf.h"

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "onnx_builder.h"

using ajp::RefusalCode;
using namespace ajp::gpu::ir;

namespace {

std::vector<std::byte> octets(std::initializer_list<unsigned> values) {
  std::vector<std::byte> out;
  for (const unsigned v : values) out.push_back(static_cast<std::byte>(v));
  return out;
}

// Lit tous les champs ; le premier refus arrête la lecture. Les champs Length
// pointent dans data : un tampon temporaire est interdit.
std::expected<std::vector<proto::Field>, ajp::Refusal> fields(std::vector<std::byte>&& data) = delete;
std::expected<std::vector<proto::Field>, ajp::Refusal> fields(const std::vector<std::byte>& data) {
  proto::Reader reader(data);
  std::vector<proto::Field> out;
  for (;;) {
    auto field = reader.next();
    if (!field) return std::unexpected(field.error());
    if (!*field) return out;
    out.push_back(**field);
  }
}

RefusalCode refusal_of(const std::vector<std::byte>& data) {
  auto read = fields(data);
  REQUIRE_FALSE(read);
  return read.error().code;
}

}  // namespace

TEST_CASE("protobuf : varint sur plusieurs octets", "[gpu_ir][protobuf]") {
  // Exemple de la documentation protobuf : champ 1 = 150.
  const std::vector<std::byte> data = octets({0x08, 0x96, 0x01});
  auto read = fields(data);
  REQUIRE(read);
  REQUIRE(read->size() == 1);
  CHECK((*read)[0].number == 1);
  CHECK((*read)[0].wire == proto::WireType::Varint);
  CHECK((*read)[0].scalar == 150);
}

TEST_CASE("protobuf : int64 négatif en complément à deux sur dix octets", "[gpu_ir][protobuf]") {
  const std::vector<std::int64_t> values{-1, std::numeric_limits<std::int64_t>::min(),
                                         std::numeric_limits<std::int64_t>::max(), 0, 300};
  test::ProtoWriter w;
  for (const std::int64_t v : values) w.varint(3, v);
  const std::vector<std::byte> data = w.data();
  auto read = fields(data);
  REQUIRE(read);
  REQUIRE(read->size() == values.size());
  for (std::size_t k = 0; k < values.size(); ++k) {
    auto value = proto::as_int64((*read)[k]);
    REQUIRE(value);
    CHECK(*value == values[k]);
  }
  // Étiquette d'un octet, puis 10 octets pour -1 et INT64_MIN, 9 pour INT64_MAX.
  CHECK(w.data().size() == 11 + 11 + 10 + 2 + 3);
}

TEST_CASE("protobuf : int32 hors bornes refusé", "[gpu_ir][protobuf]") {
  test::ProtoWriter w;
  w.varint(1, std::int64_t{1} << 31).varint(1, -(std::int64_t{1} << 31));
  const std::vector<std::byte> data = w.data();
  auto read = fields(data);
  REQUIRE(read);
  CHECK_FALSE(proto::as_int32((*read)[0]));
  auto low = proto::as_int32((*read)[1]);
  REQUIRE(low);
  CHECK(*low == std::numeric_limits<std::int32_t>::min());
}

TEST_CASE("protobuf : fixed32, fixed64 et longueur", "[gpu_ir][protobuf]") {
  test::ProtoWriter w;
  w.f32(2, 1.5F).fixed64(4, 0x0102030405060708ULL).bytes(5, "abc");
  const std::vector<std::byte> data = w.data();
  auto read = fields(data);
  REQUIRE(read);
  REQUIRE(read->size() == 3);
  auto f = proto::as_float((*read)[0]);
  REQUIRE(f);
  CHECK(*f == 1.5F);
  CHECK((*read)[1].wire == proto::WireType::Fixed64);
  CHECK((*read)[1].scalar == 0x0102030405060708ULL);
  auto s = proto::as_string((*read)[2]);
  REQUIRE(s);
  CHECK(*s == "abc");
  // Le type filaire est vérifié à la conversion.
  CHECK(proto::as_float((*read)[2]).error().code == RefusalCode::OnnxMalformed);
  CHECK_FALSE(proto::as_int64((*read)[0]));
  CHECK_FALSE(proto::as_string((*read)[1]));
}

TEST_CASE("protobuf : listes empaquetées et élément par élément", "[gpu_ir][protobuf]") {
  const std::vector<std::int64_t> ints{1, -2, 1 << 20};
  const std::vector<float> floats{0.5F, -3.0F};
  test::ProtoWriter w;
  w.packed_varints(8, ints).varint(8, 7).packed_floats(7, floats).f32(7, 9.0F);
  const std::vector<std::byte> data = w.data();
  auto read = fields(data);
  REQUIRE(read);
  REQUIRE(read->size() == 4);
  std::vector<std::int64_t> got_ints;
  std::vector<float> got_floats;
  REQUIRE(proto::append_int64s((*read)[0], got_ints));
  REQUIRE(proto::append_int64s((*read)[1], got_ints));
  REQUIRE(proto::append_floats((*read)[2], got_floats));
  REQUIRE(proto::append_floats((*read)[3], got_floats));
  CHECK(got_ints == std::vector<std::int64_t>{1, -2, 1 << 20, 7});
  CHECK(got_floats == std::vector<float>{0.5F, -3.0F, 9.0F});
}

TEST_CASE("protobuf : floats empaquetés d'une taille non multiple de 4 refusés", "[gpu_ir][protobuf]") {
  const std::vector<std::byte> data = test::ProtoWriter{}.bytes(7, "abcde").data();
  auto read = fields(data);
  REQUIRE(read);
  std::vector<float> out;
  CHECK(proto::append_floats((*read)[0], out).error().code == RefusalCode::OnnxMalformed);
}

TEST_CASE("protobuf : entiers empaquetés tronqués refusés", "[gpu_ir][protobuf]") {
  const std::vector<std::byte> data =
      octets({0x42, 0x02, 0x01, 0x80});  // champ 8, longueur 2 : 1, puis un varint coupé
  auto read = fields(data);
  REQUIRE(read);
  std::vector<std::int64_t> out;
  CHECK(proto::append_int64s((*read)[0], out).error().code == RefusalCode::OnnxMalformed);
}

TEST_CASE("protobuf : messages tronqués ou incohérents refusés", "[gpu_ir][protobuf]") {
  CHECK(refusal_of(octets({0x08})) == RefusalCode::OnnxMalformed);                       // valeur absente
  CHECK(refusal_of(octets({0x08, 0x80})) == RefusalCode::OnnxMalformed);                 // varint coupé
  CHECK(refusal_of(octets({0x12, 0x05, 0x61})) == RefusalCode::OnnxMalformed);           // longueur au-delà
  CHECK(refusal_of(octets({0x15, 0x01, 0x02})) == RefusalCode::OnnxMalformed);           // fixed32 coupé
  CHECK(refusal_of(octets({0x09, 1, 2, 3, 4, 5, 6, 7})) == RefusalCode::OnnxMalformed);  // fixed64 coupé
  CHECK(refusal_of(octets({0x00, 0x01})) == RefusalCode::OnnxMalformed);                 // champ 0
  CHECK(refusal_of(octets({0x0B})) == RefusalCode::OnnxMalformed);                       // groupe (type 3)
  CHECK(refusal_of(octets({0x0C})) == RefusalCode::OnnxMalformed);        // fin de groupe (type 4)
  CHECK(refusal_of(octets({0x0E, 0x00})) == RefusalCode::OnnxMalformed);  // type filaire 6
  // Varint de 11 octets, et dixième octet au-delà de 64 bits.
  CHECK(refusal_of(octets({0x08, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x80, 0x01})) ==
        RefusalCode::OnnxMalformed);
  CHECK(refusal_of(octets({0x08, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x02})) ==
        RefusalCode::OnnxMalformed);
  // Longueur énorme : refusée sans débordement du calcul de position.
  CHECK(refusal_of(octets({0x12, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01})) ==
        RefusalCode::OnnxMalformed);
}

TEST_CASE("protobuf : message vide et numéro de champ maximal", "[gpu_ir][protobuf]") {
  const std::vector<std::byte> none;
  auto empty = fields(none);
  REQUIRE(empty);
  CHECK(empty->empty());
  test::ProtoWriter w;
  w.tag((1U << 29U) - 1U, 0).raw_varint(4);
  const std::vector<std::byte> data = w.data();
  auto read = fields(data);
  REQUIRE(read);
  CHECK((*read)[0].number == (1U << 29U) - 1U);
}
