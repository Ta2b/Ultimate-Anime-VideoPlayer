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

#include "core/sha256_blocks.h"
#include "prepare/utf8.h"

namespace ajp::prepare {
void Sha256::blocks(const std::uint8_t* data, std::size_t count) {
  core::sha256_blocks(state_, data, count);
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
