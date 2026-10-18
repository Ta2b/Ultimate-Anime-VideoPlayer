// Repris de animejanai-inference-rtx@118c272 src/content_identity.h:16-80.
#include "prepare/sha256.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <format>
#include <fstream>
#include <vector>

namespace ajp::prepare {
namespace {

constexpr std::array<std::uint32_t, 64> kRound = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

}  // namespace

void Sha256::block(const std::uint8_t* data) {
  std::array<std::uint32_t, 64> w{};
  for (std::size_t i = 0; i < 16; ++i) {
    w[i] = std::uint32_t{data[4 * i]} << 24U | std::uint32_t{data[4 * i + 1]} << 16U |
           std::uint32_t{data[4 * i + 2]} << 8U | std::uint32_t{data[4 * i + 3]};
  }
  for (std::size_t i = 16; i < 64; ++i) {
    const std::uint32_t a = w[i - 15];
    const std::uint32_t b = w[i - 2];
    w[i] = w[i - 16] + (std::rotr(a, 7) ^ std::rotr(a, 18) ^ (a >> 3U)) + w[i - 7] +
           (std::rotr(b, 17) ^ std::rotr(b, 19) ^ (b >> 10U));
  }
  auto s = state_;
  for (std::size_t i = 0; i < 64; ++i) {
    const std::uint32_t first = s[7] + (std::rotr(s[4], 6) ^ std::rotr(s[4], 11) ^ std::rotr(s[4], 25)) +
                                ((s[4] & s[5]) ^ (~s[4] & s[6])) + kRound[i] + w[i];
    const std::uint32_t second = (std::rotr(s[0], 2) ^ std::rotr(s[0], 13) ^ std::rotr(s[0], 22)) +
                                 ((s[0] & s[1]) ^ (s[0] & s[2]) ^ (s[1] & s[2]));
    for (std::size_t j = 7; j > 0; --j) s[j] = s[j - 1];
    s[4] += first;
    s[0] = first + second;
  }
  for (std::size_t i = 0; i < 8; ++i) state_[i] += s[i];
}

void Sha256::update(std::span<const std::byte> bytes) {
  bytes_ += bytes.size();
  while (!bytes.empty()) {
    const std::size_t n = std::min(bytes.size(), buffer_.size() - used_);
    std::memcpy(buffer_.data() + used_, bytes.data(), n);
    used_ += n;
    bytes = bytes.subspan(n);
    if (used_ == buffer_.size()) {
      block(buffer_.data());
      used_ = 0;
    }
  }
}

void Sha256::update(std::string_view bytes) {
  update(std::as_bytes(std::span(bytes.data(), bytes.size())));
}

void Sha256::integer(std::uint64_t value) {
  std::array<std::byte, 8> data{};
  for (std::size_t i = 0; i < 8; ++i) data[7 - i] = static_cast<std::byte>(value >> (8 * i));
  update(data);
}

void Sha256::text(std::string_view value) {
  integer(value.size());
  update(value);
}

std::string Sha256::hex() const {
  Sha256 copy = *this;
  const std::uint64_t bits = bytes_ * 8;
  copy.update(std::array{std::byte{0x80}});
  while (copy.used_ != 56) copy.update(std::array{std::byte{0}});
  copy.integer(bits);
  std::string result;
  result.reserve(64);
  for (const std::uint32_t word : copy.state_) result += std::format("{:08x}", word);
  return result;
}

std::string sha256_hex(std::string_view bytes) {
  Sha256 hash;
  hash.update(bytes);
  return hash.hex();
}

std::expected<std::string, Refusal> sha256_file(const std::filesystem::path& file) {
  std::ifstream stream(file, std::ios::binary);
  if (!stream) return std::unexpected(Refusal{RefusalCode::CacheIo, "lecture impossible : " + file.string()});
  Sha256 hash;
  std::vector<char> buffer(1 << 16);
  while (stream) {
    stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    hash.update(std::string_view(buffer.data(), static_cast<std::size_t>(stream.gcount())));
  }
  if (stream.bad())
    return std::unexpected(Refusal{RefusalCode::CacheIo, "lecture interrompue : " + file.string()});
  return hash.hex();
}

bool is_sha256_hex(std::string_view text) {
  return text.size() == 64 &&
         std::ranges::all_of(text, [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

}  // namespace ajp::prepare
