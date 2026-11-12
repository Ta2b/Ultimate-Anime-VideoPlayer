// Les deux chemins de SHA-256 (instructions SHA du processeur, code portable) donnent le même résultat : sur
// une machine à instructions SHA, les vecteurs de model_index_test ne passent que par le premier (C06-11).
#include "core/sha256_blocks.h"

#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using namespace ajp::core;

namespace {

constexpr std::array<std::uint32_t, 8> kInitial = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

}  // namespace

TEST_CASE("SHA-256 : le chemin portable donne le vecteur « abc » de FIPS 180-2", "[sha256]") {
  // « abc » bourré à un bloc : 0x80 puis la longueur en bits (24) sur 8 octets.
  std::array<std::uint8_t, 64> block{};
  block[0] = 'a';
  block[1] = 'b';
  block[2] = 'c';
  block[3] = 0x80;
  block[63] = 24;
  auto state = kInitial;
  sha256_blocks_portable(state, block.data(), 1);
  const std::array<std::uint32_t, 8> expected = {0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223,
                                                 0xb00361a3, 0x96177a9c, 0xb410ff61, 0xf20015ad};
  CHECK(state == expected);
}

TEST_CASE("SHA-256 : le chemin du processeur et le chemin portable s'accordent", "[sha256]") {
  std::mt19937_64 random(20261112);
  for (const std::size_t blocks : {1U, 2U, 3U, 17U, 1000U}) {
    std::vector<std::uint8_t> data(blocks * 64);
    for (std::uint8_t& byte : data) byte = static_cast<std::uint8_t>(random());
    auto native = kInitial;
    auto portable = kInitial;
    sha256_blocks(native, data.data(), blocks);
    sha256_blocks_portable(portable, data.data(), blocks);
    CHECK(native == portable);
  }
  // Un état de départ quelconque, pas seulement l'initial.
  std::array<std::uint32_t, 8> start{};
  for (std::uint32_t& word : start) word = static_cast<std::uint32_t>(random());
  std::vector<std::uint8_t> data(64 * 5);
  for (std::uint8_t& byte : data) byte = static_cast<std::uint8_t>(random());
  auto native = start;
  auto portable = start;
  sha256_blocks(native, data.data(), 5);
  sha256_blocks_portable(portable, data.data(), 5);
  CHECK(native == portable);
}
