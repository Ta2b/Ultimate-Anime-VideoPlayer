// Repris de animejanai-inference-rtx@118c272 src/content_identity.h:16-80.
#include "prepare/sha256.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <format>
#include <fstream>
#include <map>
#include <mutex>
#include <system_error>
#include <vector>

#include "prepare/utf8.h"

#if defined(__x86_64__) || defined(_M_X64)
#define AJP_SHA_NI 1
#include <immintrin.h>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

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

void block_portable(std::array<std::uint32_t, 8>& state, const std::uint8_t* data) {
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
  auto s = state;
  for (std::size_t i = 0; i < 64; ++i) {
    const std::uint32_t first = s[7] + (std::rotr(s[4], 6) ^ std::rotr(s[4], 11) ^ std::rotr(s[4], 25)) +
                                ((s[4] & s[5]) ^ (~s[4] & s[6])) + kRound[i] + w[i];
    const std::uint32_t second = (std::rotr(s[0], 2) ^ std::rotr(s[0], 13) ^ std::rotr(s[0], 22)) +
                                 ((s[0] & s[1]) ^ (s[0] & s[2]) ^ (s[1] & s[2]));
    for (std::size_t j = 7; j > 0; --j) s[j] = s[j - 1];
    s[4] += first;
    s[0] = first + second;
  }
  for (std::size_t i = 0; i < 8; ++i) state[i] += s[i];
}

#ifdef AJP_SHA_NI
// P4.10 : les instructions SHA des x86 (Intel depuis Goldmont et Ice Lake, AMD depuis Zen), 4 à 5 fois le
// débit du code portable. Quatre tours par groupe ; le message des groupes 4 à 15 se calcule au fil de l'eau.
bool cpu_has_sha() {
#ifdef _MSC_VER
  std::array<int, 4> r{};
  __cpuid(r.data(), 1);
  const bool sse41 = (r[2] & (1 << 19)) != 0 && (r[2] & (1 << 9)) != 0;
  __cpuidex(r.data(), 7, 0);
  return sse41 && (r[1] & (1 << 29)) != 0;
#else
  unsigned a = 0, b = 0, c = 0, d = 0;
  if (__get_cpuid(1, &a, &b, &c, &d) == 0 || (c & (1U << 19)) == 0 || (c & (1U << 9)) == 0) return false;
  if (__get_cpuid_count(7, 0, &a, &b, &c, &d) == 0) return false;
  return (b & (1U << 29)) != 0;
#endif
}

#ifndef _MSC_VER
__attribute__((target("sha,sse4.1,ssse3")))
#endif
void blocks_sha_ni(std::array<std::uint32_t, 8>& state, const std::uint8_t* data, std::size_t count) {
  const __m128i mask = _mm_set_epi64x(0x0c0d0e0f08090a0bLL, 0x0405060700010203LL);
  __m128i tmp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&state[0]));
  __m128i state1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&state[4]));
  tmp = _mm_shuffle_epi32(tmp, 0xB1);                // CDAB
  state1 = _mm_shuffle_epi32(state1, 0x1B);          // EFGH
  __m128i state0 = _mm_alignr_epi8(tmp, state1, 8);  // ABEF
  state1 = _mm_blend_epi16(state1, tmp, 0xF0);       // CDGH
  for (; count > 0; --count, data += 64) {
    const __m128i abef = state0;
    const __m128i cdgh = state1;
    __m128i msg[4];  // NOLINT(modernize-avoid-c-arrays) : std::array perd les attributs de __m128i
    for (std::size_t i = 0; i < 4; ++i)
      msg[i] = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(data + 16 * i)), mask);
    for (std::size_t g = 0; g < 16; ++g) {
      const __m128i k = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&kRound[4 * g]));
      __m128i t = _mm_add_epi32(msg[g & 3U], k);
      state1 = _mm_sha256rnds2_epu32(state1, state0, t);
      t = _mm_shuffle_epi32(t, 0x0E);
      state0 = _mm_sha256rnds2_epu32(state0, state1, t);
      if (g < 12) {
        __m128i next = _mm_sha256msg1_epu32(msg[g & 3U], msg[(g + 1) & 3U]);
        next = _mm_add_epi32(next, _mm_alignr_epi8(msg[(g + 3) & 3U], msg[(g + 2) & 3U], 4));
        msg[g & 3U] = _mm_sha256msg2_epu32(next, msg[(g + 3) & 3U]);
      }
    }
    state0 = _mm_add_epi32(state0, abef);
    state1 = _mm_add_epi32(state1, cdgh);
  }
  tmp = _mm_shuffle_epi32(state0, 0x1B);        // FEBA
  state1 = _mm_shuffle_epi32(state1, 0xB1);     // DCHG
  state0 = _mm_blend_epi16(tmp, state1, 0xF0);  // DCBA
  state1 = _mm_alignr_epi8(state1, tmp, 8);     // ABEF
  _mm_storeu_si128(reinterpret_cast<__m128i*>(&state[0]), state0);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(&state[4]), state1);
}
#endif

}  // namespace

void Sha256::blocks(const std::uint8_t* data, std::size_t count) {
#ifdef AJP_SHA_NI
  static const bool sha_ni = cpu_has_sha();
  if (sha_ni) {
    blocks_sha_ni(state_, data, count);
    return;
  }
#endif
  for (; count > 0; --count, data += 64) block_portable(state_, data);
}

void Sha256::update(std::span<const std::byte> bytes) {
  bytes_ += bytes.size();
  if (used_ > 0) {  // le bloc entamé d'abord
    const std::size_t n = std::min(bytes.size(), buffer_.size() - used_);
    std::memcpy(buffer_.data() + used_, bytes.data(), n);
    used_ += n;
    bytes = bytes.subspan(n);
    if (used_ < buffer_.size()) return;
    blocks(buffer_.data(), 1);
    used_ = 0;
  }
  // Les blocs complets se lisent en place, sans passer par le tampon (P4.10).
  const std::size_t whole = bytes.size() / buffer_.size();
  if (whole > 0) blocks(reinterpret_cast<const std::uint8_t*>(bytes.data()), whole);
  bytes = bytes.subspan(whole * buffer_.size());
  std::memcpy(buffer_.data(), bytes.data(), bytes.size());
  used_ = bytes.size();
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
  if (!stream)
    return std::unexpected(Refusal{RefusalCode::CacheIo, "lecture impossible : " + utf8_path(file)});
  Sha256 hash;
  std::vector<char> buffer(1 << 16);
  while (stream) {
    stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    hash.update(std::string_view(buffer.data(), static_cast<std::size_t>(stream.gcount())));
  }
  if (stream.bad())
    return std::unexpected(Refusal{RefusalCode::CacheIo, "lecture interrompue : " + utf8_path(file)});
  return hash.hex();
}

std::expected<std::string, Refusal> sha256_file_known(const std::filesystem::path& file) {
  struct Known {
    std::uintmax_t size = 0;
    std::filesystem::file_time_type modified;
    std::string digest;
  };
  static std::mutex mutex;
  static std::map<std::filesystem::path, Known> known;
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(file, error);
  const std::filesystem::file_time_type modified =
      error ? std::filesystem::file_time_type{} : std::filesystem::last_write_time(file, error);
  if (error) return sha256_file(file);  // son refus, avec le chemin
  {
    const std::scoped_lock lock(mutex);
    if (const auto found = known.find(file);
        found != known.end() && found->second.size == size && found->second.modified == modified)
      return found->second.digest;
  }
  auto digest = sha256_file(file);
  if (!digest) return digest;
  const std::scoped_lock lock(mutex);
  known.insert_or_assign(file, Known{.size = size, .modified = modified, .digest = *digest});
  return digest;
}

bool is_sha256_hex(std::string_view text) {
  return text.size() == 64 &&
         std::ranges::all_of(text, [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

}  // namespace ajp::prepare
