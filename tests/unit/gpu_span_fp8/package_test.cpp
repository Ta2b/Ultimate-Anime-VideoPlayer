// Le paquet de poids du SPAN FP8 : aller-retour, refus d'un fichier abîmé, tailles prises (sans GPU).
#include "gpu/span_fp8/package.h"

#include <cstring>
#include <limits>

#include <catch2/catch_test_macros.hpp>

namespace ajp::gpu::span_fp8 {
namespace {

std::vector<std::byte> valid_bytes() {
  return serialize(synthetic());
}

}  // namespace

TEST_CASE("un paquet écrit se relit à l'identique", "[span_fp8]") {
  Package original = synthetic();
  original.scales[3] = 0.125F;
  original.blocks[7].attention.bias[63] = -2.5F;
  original.pixel.weights.back() = 0x3C00;  // 1,0
  const std::vector<std::byte> bytes = serialize(original);
  const auto back = parse(bytes);
  REQUIRE(back);
  CHECK(back->scales == original.scales);
  CHECK(back->first.weights == original.first.weights);
  CHECK(back->blocks[7].attention.bias == original.blocks[7].attention.bias);
  CHECK(back->blocks[0].conv[2].weights.size() == std::size_t{64} * 9 * 64);
  CHECK(back->concat.weights.size() == std::size_t{64} * 256);
  CHECK(back->pixel.weights == original.pixel.weights);
  CHECK(back->pixel.bias.size() == 48U);
}

TEST_CASE("un paquet tronqué, abîmé ou d'une autre version est refusé", "[span_fp8]") {
  const std::vector<std::byte> good = valid_bytes();
  REQUIRE(parse(good));

  CHECK_FALSE(parse({}));
  CHECK_FALSE(parse(std::span(good).first(good.size() - 1)));
  CHECK_FALSE(parse(std::span(good).first(kHeaderBytes)));

  std::vector<std::byte> flipped = good;
  flipped[good.size() / 2] ^= std::byte{0x10};  // un bit change : la somme de contrôle le voit
  const auto crc = parse(flipped);
  REQUIRE_FALSE(crc);
  CHECK(crc.error().code == RefusalCode::FileUnreadable);

  std::vector<std::byte> magic = good;
  magic[0] = std::byte{'X'};
  CHECK(parse(magic).error().code == RefusalCode::FormatUnsupported);

  std::vector<std::byte> version = good;
  version[8] = std::byte{2};
  CHECK(parse(version).error().code == RefusalCode::FormatUnsupported);

  std::vector<std::byte> blocks = good;
  blocks[12] = std::byte{5};  // 5 blocs : pas ce SPAN
  CHECK(parse(blocks).error().code == RefusalCode::FormatUnsupported);
}

TEST_CASE("une échelle nulle ou un poids infini rend le paquet inutilisable", "[span_fp8]") {
  Package zero = synthetic();
  zero.scales[10] = 0.0F;
  CHECK_FALSE(parse(serialize(zero)));

  Package inf = synthetic();
  inf.blocks[2].conv[1].weights[5] = 0x7C00;  // +infini
  CHECK_FALSE(parse(serialize(inf)));

  Package nan = synthetic();
  nan.first.bias[0] = std::numeric_limits<float>::quiet_NaN();
  CHECK_FALSE(parse(serialize(nan)));
}

TEST_CASE("le CRC-32 est celui de zlib", "[span_fp8]") {
  constexpr char kText[] = "123456789";
  std::vector<std::byte> bytes(9);
  std::memcpy(bytes.data(), kText, 9);
  CHECK(crc32(bytes) == 0xCBF43926U);  // valeur de contrôle classique du CRC-32 IEEE
  CHECK(crc32({}) == 0U);
}

TEST_CASE("les tailles que les noyaux prennent", "[span_fp8]") {
  CHECK(size_supported(1088, 1920));  // la taille de travail de 1080p
  CHECK(size_supported(1080, 1920));
  CHECK(size_supported(720, 1280));
  CHECK(size_supported(2160, 3840));
  CHECK(size_refusal(1088, 1920).empty());
  CHECK_FALSE(size_supported(540, 960));  // 129 600 pixels de caractéristiques, pas un multiple de 128
  CHECK_FALSE(size_supported(1087, 1920));
  CHECK_FALSE(size_supported(1088, 1919));
  CHECK_FALSE(size_supported(0, 1920));
  CHECK_FALSE(size_supported(-2, 1920));
  CHECK_FALSE(size_supported(1088, 40000));
  CHECK(size_refusal(540, 960).find("128") != std::string::npos);
}

}  // namespace ajp::gpu::span_fp8
